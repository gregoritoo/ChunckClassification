import torch
from transformers import RobertaForSequenceClassification
from transformers import BertTokenizer, RobertaTokenizer, BertweetTokenizer
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm


class SentimentModel(nn.Module):
    def __init__(self, device, num_labels=3):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis",
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.tokenizer = BertweetTokenizer.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis", do_lower_case=True
        )
        self.device = device

    def encode_batch(self, data, labels, batch_size=2):
        encoded_data_train = self.tokenizer.batch_encode_plus(
            data,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            max_length=100,
            return_tensors="pt",
            truncation=True,
        )
        input_ids_train = encoded_data_train["input_ids"]
        attention_masks_train = encoded_data_train["attention_mask"]
        labels_train = torch.tensor(labels)
        dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=batch_size
        )
        return dataloader

    def compute_balance_accuracy(self, true_vals, predictions, nb_class=3):
        nb_class = 3
        accs = [0 for _ in range(nb_class)]
        for classe in range(nb_class):
            accs[classe] = np.sum(
                (true_vals == np.argmax(predictions, axis=-1))
                * (true_vals == classe).astype(int)
            ) / max((true_vals == 2).sum(), 1)
        return np.mean(np.array(accs))

    def evaluate(self, dataloader):
        self.model.eval()
        loss_val_total = 0
        predictions, true_vals = [], []
        for batch in tqdm(dataloader):
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        loss_val_avg = loss_val_total / len(dataloader)
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train(self, dataloader_train, dataloader_val, optimizer, scheduler, epochs=10):
        training_losses = []
        test_losses = []
        testing_accuracy = []
        training_accuracy = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc="Epoch {:1d}".format(epoch))
            for batch in progress_bar:
                self.model.zero_grad()
                batch = tuple(b.to(self.device) for b in batch)
                inputs = {"input_ids": batch[0], "labels": batch[2]}

                outputs = self.model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix(
                    {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
                )

                val_loss, predictions, true_vals = self.evaluate(dataloader_train)
                training_acc = self.compute_balance_accuracy(true_vals, predictions)
                print(f"Training accuracy : {training_acc}")

            tqdm.write(f"\nEpoch {epoch}")

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f"Training loss: {loss_train_avg}")

            val_loss, predictions, true_vals = self.evaluate(dataloader_val)
            balanced_acc = self.compute_balance_accuracy(true_vals, predictions)
            tqdm.write(f"Validation loss: {val_loss}")
            tqdm.write(f"balanced_acc: {balanced_acc}")
            training_losses.append(loss_train_avg)
            test_losses.append(val_loss)
            testing_accuracy.append(val_f1)
            training_accuracy.append(training_acc)

    def get_scores(self, text, encode=True):
        outputs = self.predict(text, encode=encode)
        scores = torch.nn.Softmax(dim=1)(outputs.logits)
        return scores

    def predict(self, text, encode=True):
        if encode == True:
            encoded_sentence = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                return_attention_mask=True,
                padding=True,
                max_length=100,
                return_tensors="pt",
                truncation=True,
            )
        else:
            encoded_sentence = text
        input_ids = encoded_sentence.to(self.device)
        self.model.eval()
        inputs = {
            "input_ids": input_ids,
            #'labels': labels_train,
        }

        outputs = self.model(**inputs)
        outputs.logits = torch.nn.Softmax(dim=1)(outputs.logits)
        return outputs

    def load_pretrained(self, path="../models/roberta_sentiment_classification"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_embeddings(self, text):
        outputs = self.predict(text)
        embeddings = torch.concat(outputs.hidden_states[-4:], dim=-1)
        return embeddings.mean(dim=1)
