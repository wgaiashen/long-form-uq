# the code adapted from https://github.com/yuh-zha/AlignScore
import os
import subprocess
import sys
import spacy
from typing import Optional, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    RobertaModel,
    RobertaForMaskedLM,
)
import nltk
import torch
import torch.nn as nn
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from logging import warning
from typing import List
from tqdm import tqdm

class AlignScorer:
    def __init__(
        self,
        model: str,
        batch_size: int,
        device: int,
        ckpt_path: str,
        evaluation_mode="nli_sp",
        verbose=True,
    ) -> None:
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
                    "--retries",
                    "1",
                    "--timeout",
                    "1",
                    "-q",
                ]
            )
        self.model = Inferencer(
            ckpt_path=ckpt_path,
            model=model,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        nltk.download("punkt")
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.nlg_eval(contexts, claims)[1].tolist()


class Inferencer:
    def __init__(
        self,
        ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",  # added direct url from huggingface
        model="bert-base-uncased",
        batch_size=32,
        device="cuda",
        verbose=True,
    ) -> None:
        self.device = device
        if ckpt_path is not None:
            self.model = BERTAlignModel(model=model)
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path)["state_dict"]
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    ckpt_path, progress=False
                )["state_dict"]

            self.model.load_state_dict(state_dict, strict=False)

            # [vendored-patch] guard GPU-only cache ops so this also runs on a CPU node
            # (the original called these unconditionally and crashed with "no NVIDIA driver"
            # on the login node). No effect on scoring; GPU path is unchanged.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.model = self.model.to(self.device)
        else:
            warning("loading UNTRAINED model!")
            self.model = BERTAlignModel(model=model).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.spacy = spacy.load("en_core_web_sm")

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=-1)

        self.disable_progress_bar_in_inference = False

        self.nlg_eval_mode = None  # bin, bin_sp, nli, nli_sp
        self.verbose = verbose

    def inference_example_batch(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SummaC Style aggregation
        """
        self.disable_progress_bar_in_inference = True
        assert len(premise) == len(
            hypo
        ), "Premise must has the same length with Hypothesis!"

        out_score = []
        for one_pre, one_hypo in tqdm(
            zip(premise, hypo),
            desc="Evaluating",
            total=len(premise),
            disable=(not self.verbose),
        ):
            out_score.append(self.inference_per_example(one_pre, one_hypo))

        return None, torch.tensor(out_score), None

    def inference_per_example(self, premise: str, hypo: str):
        """
        inference a example,
        premise: string
        hypo: string
        using self.inference to batch the process
        """

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield " ".join(lst[i : i + n])

        premise_sents = sent_tokenize(premise)
        premise_sents = premise_sents or [""]

        n_chunk = len(premise.strip().split()) // 350 + 1
        n_chunk = max(len(premise_sents) // n_chunk, 1)
        premise_sents = [each for each in chunks(premise_sents, n_chunk)]

        hypo_sents = sent_tokenize(hypo)

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])

        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == "nli_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][
                    :, 0
                ]  # use NLI head OR ALIGN head
            elif self.nlg_eval_mode == "bin_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[
                    1
                ]  # use NLI head OR ALIGN head
            elif self.nlg_eval_mode == "reg_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[
                    0
                ]  # use NLI head OR ALIGN head

            output_score = (
                output_score.view(len(premise_sents), len(hypo_sents))
                .max(dim=0)
                .values.mean()
                .item()
            )  # sum or mean depends on the task/aspect
            return output_score

        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][
            :, 0
        ]  # use NLI head OR ALIGN head
        output_score = (
            output_score.view(len(premise_sents), len(hypo_sents))
            .max(dim=0)
            .values.mean()
            .item()
        )  # sum or mean depends on the task/aspect

        return output_score

    def inference(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]

        batch = self.batch_tokenize(premise, hypo)
        output_score_reg = []
        output_score_bin = []
        output_score_tri = []

        for mini_batch in tqdm(
            batch,
            desc="Evaluating",
            disable=not self.verbose or self.disable_progress_bar_in_inference,
        ):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch)
                model_output_reg = model_output.reg_label_logits.cpu()
                model_output_bin = (
                    model_output.seq_relationship_logits
                )  # Temperature Scaling / 2.5
                model_output_tri = model_output.tri_label_logits

                model_output_bin = self.softmax(model_output_bin).cpu()
                model_output_tri = self.softmax(model_output_tri).cpu()
            output_score_reg.append(model_output_reg[:, 0])
            output_score_bin.append(model_output_bin[:, 1])
            output_score_tri.append(model_output_tri[:, :])

        output_score_reg = torch.cat(output_score_reg)
        output_score_bin = torch.cat(output_score_bin)
        output_score_tri = torch.cat(output_score_tri)

        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == "nli":
                output_score_nli = output_score_tri[:, 0]
                return None, output_score_nli, None
            elif self.nlg_eval_mode == "bin":
                return None, output_score_bin, None
            elif self.nlg_eval_mode == "reg":
                return None, output_score_reg, None
            else:
                ValueError("unrecognized nlg eval mode")

        return output_score_reg, output_score_bin, output_score_tri

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(
            hypo
        ), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(
            self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)
        ):
            try:
                mini_batch = self.tokenizer(
                    mini_batch_pre,
                    mini_batch_hypo,
                    truncation="only_first",
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
            except Exception as exception:
                warning(f"text_b too long... error: {exception}")
                mini_batch = self.tokenizer(
                    mini_batch_pre,
                    mini_batch_hypo,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
            batch.append(mini_batch)

        return batch

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def nlg_eval(self, premise, hypo):
        assert self.nlg_eval_mode is not None, "Select NLG Eval mode!"
        if (
            (self.nlg_eval_mode == "bin")
            or (self.nlg_eval_mode == "nli")
            or (self.nlg_eval_mode == "reg")
        ):
            return self.inference(premise, hypo)

        elif (
            (self.nlg_eval_mode == "bin_sp")
            or (self.nlg_eval_mode == "nli_sp")
            or (self.nlg_eval_mode == "reg_sp")
        ):
            return self.inference_example_batch(premise, hypo)

        else:
            ValueError("Unrecognized NLG Eval mode!")


class BERTAlignModel(nn.Module):  # changed pytorch_lightning to pytorch
    def __init__(
        self, model="roberta-large", using_pretrained=True, *args, **kwargs
    ) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.model = model

        if "roberta" in model:
            if using_pretrained:
                self.base_model = RobertaModel.from_pretrained(model)
                self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
            else:
                self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
                self.mlm_head = RobertaForMaskedLM(
                    AutoConfig.from_pretrained(model)
                ).lm_head

        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        base_model_output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=(
                batch["token_type_ids"] if "token_type_ids" in batch.keys() else None
            ),
        )

        prediction_scores = self.mlm_head(
            base_model_output.last_hidden_state
        )  # sequence_output for mlm
        seq_relationship_score = self.bin_layer(
            self.dropout(base_model_output.pooler_output)
        )  # pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        return ModelOutput(
            loss=None,
            all_loss=None,
            loss_nums=None,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )


@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
