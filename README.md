# EN→HI Machine Translation (XLM‑R encoder + TensorFlow decoder)

This README explains the full pipeline used in the provided script, from preprocessing to training and testing. It also includes the core math behind each component and references the exact function/method names in the code where those equations are implemented.

> Pipeline components: **TensorFlow/Keras** (decoder + training), **Hugging Face** (XLM‑R encoder + tokenizers), **datasets** (IITB EN–HI), **tokenizers** (fast BPE for Hindi).

---

## 1) Dataset & Splits

* **Source**: `cfilt/iitb-english-hindi` via IIT Bombay English to hindi `datasets` on huggingface.
* **Splits**: `train`, `validation`, `test`.
* **Fields**: `ds['split']['translation']['en']` and `['hi']`.

**Code entry points**

* Dataset is loaded  `ds = load_dataset("cfilt/iitb-english-hindi")``.

---

## 2) Tokenization

### 2.1 Source (English) — XLM‑R tokenizer

* Uses the pretrained XLM‑RoBERTa tokenizer.
* Produces `input_ids` and `attention_mask` with fixed length `max_source_length` via truncation/padding.

**Code**: `tf_tokenize_en_batch`, called internally by `make_ds`.

**Shapes**: `(batch, src_len)` for both `input_ids` and `attention_mask`.

### 2.2 Target (Hindi) — Fast BPE tokenizer

* Trained with huggingface `tokenizers` (Rust backend), vocab size ≈ 16k.
* Special tokens: `[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`.
* For each Hindi sentence we build:

  * `decoder_inputs` = `[BOS] + tokens[:-1]`
  * `labels` = `tokens[1:] + [EOS]` (effectively a 1-position shift)

**Code**: `encode_hi_ids` → `tf_encode_hi` → used inside `make_ds`.

---

## 3) tf.data Pipeline

* `make_ds(src_texts, tgt_texts, shuffle)`:

  1. `tf_encode_hi` converts Hindi strings → ids via `tf.numpy_function`.
  2. Creates `(decoder_inputs, labels)` by shifting.
  3. `padded_batch` to fixed lengths (padding value = `pad_id`).
  4. `tf_tokenize_en_batch` tokenizes English **per batch** (vectorized) to get `input_ids` and `attention_mask`.

**Outputs per batch**

* `x = {"input_ids": (B,S), "attention_mask": (B,S), "dec_inp": (B,T)}`
* `y = labels: (B,T)`

---

## 4) Encoder (XLM‑RoBERTa, frozen)

* Model: `TFXLMRobertaModel`.
* We use only `last_hidden_state`.
* **Output shape**: `(batch, src_len, d_model)` where `d_model = 768` for XLM‑R base.

**Code**: `EncDec.call` uses `enc_xlmr().last_hidden_state`.

---

## 5) Decoder Architecture (Transformer Decoder)

The decoder stacks `num_layers` identical blocks, each with:

1. **Masked self‑attention** (causal) over target prefix
2. **Cross‑attention** over encoder outputs
3. **Position‑wise feed‑forward** network

### 5.1 Token Embedding

**Code**: `TokenEmbedding.call`

**Equation**

$$
E_t = \mathrm{Embed}(\text{token}_t) \cdot \sqrt{d_{model}}\
]
$$
where `d_model` is the hidden size.

**Shape**: `(batch, tgt_len, d_model)`.

### 5.2 Positional Encoding
**Code**: `PositionalEncoding.call`

**Equation** (sinusoidal):

$$
\mathrm{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
\mathrm{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$


Applied as:

$$
X = E + \mathrm{PE}
$$

with shape `(batch, tgt_len, d_model)`.

### 5.3 Look‑ahead Mask (causal)

**Code**: `look_ahead_mask`, combined in `TransformerDecoder.call`.

**Equation** (lower‑triangular):

$$
M_{\text{LA}}[t,u] = \begin{cases}1 & u \le t \\ 0 & u > t\end{cases}
$$

We also apply a target non‑pad mask and an encoder non‑pad mask to build the full attention masks passed to Keras `MultiHeadAttention`.

### 5.4 Multi‑Head Attention

**Code**: `DecoderLayer.call` via `tf.keras.layers.MultiHeadAttention`.

**Scaled dot‑product attention** per head:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \log M\right) V\,.
$$

`M` is the binary mask (1=allow, 0=block), injected as large negative bias on disallowed positions. For self‑attention `Q=K=V` from the decoder states; for cross‑attention `Q` from decoder and `K,V` from encoder outputs.

**Multi‑head**:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)W^O\,.
$$

### 5.5 Feed‑Forward Network

**Code**: `DecoderLayer.__init__` → `self.ffn`

**Equation**:

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2\,.
$$

### 5.6 Residual + LayerNorm

Each sub‑block is wrapped with residual and layer normalization:

$$
\mathrm{LN}(x + \mathrm{SubBlock}(x))
$$

**Code**: `DecoderLayer.call` (`n1`, `n2`, `n3` are the LayerNorms, with residual adds inline).

### 5.7 Output Projection

**Code**: `TransformerDecoder.__init__` → `self.proj`

**Equation**: logits over vocab

$$
Z = Y W_{\text{vocab}} + b,\quad Z \in \mathbb{R}^{B \times T \times |V|}
$$

---

## 6) Masks in Detail

* **Target padding mask**: `tgt_valid[b,t] = 1` if `dec_inp[b,t] != pad_id` else 0.
* **Look‑ahead mask**: lower‑triangular `(T,T)`.
* **Self‑attention mask**: elementwise min of broadcast(target mask) and look‑ahead.
* **Cross‑attention mask**: outer‑product of target valid `(B,T)` and encoder valid `(B,S)`.

**Code**: all built inside `TransformerDecoder.call`.

---

## 7) Full Forward Pass

**Code**: `EncDec.call`

1. Encode English: `enc_out = enc_xlmr(input_ids, attention_mask).last_hidden_state` → `(B,S,d_model)`
2. Build decoder masks from `dec_inp` and `attention_mask`.
3. Run decoder stack on `dec_inp` with cross‑attention to `enc_out`.
4. Project to vocabulary: logits `(B,T,|V|)`.

---

## 8) Loss & Metrics

### 8.1 Masked Cross‑Entropy Loss

**Code**: `masked_loss`

**Equation** (sparse CE, masked):

$$
\mathcal{L} = \frac{\sum_{b,t} \mathbf{1}[y_{b,t} \ne \text{PAD}] \cdot \mathrm{CE}(y_{b,t}, Z_{b,t,:})}{\sum_{b,t} \mathbf{1}[y_{b,t} \ne \text{PAD}]}\,.
$$

`Z` are logits from the decoder, `y` are integer labels.

### 8.2 Masked Token Accuracy

**Code**: `masked_acc`

**Equation**:

$$
\mathrm{Acc} = \frac{\sum_{b,t} \mathbf{1}[y_{b,t} \ne \text{PAD}] \cdot \mathbf{1}[\arg\max Z_{b,t,:} = y_{b,t}] }{\sum_{b,t} \mathbf{1}[y_{b,t} \ne \text{PAD}] }\,.
$$

---

## 9) Learning Rate Schedule (Noam)

**Code**: `Noam.__call__`

**Equation**:

$$
\mathrm{lr}(\text{step}) = \text{factor} \cdot d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})\,.
$$

Warmup stabilizes early training; then inverse‑sqrt decay.

---

## 10) Decoding (Inference)

### 10.1 Greedy Decoding

**Code**: `translate_greedy`

Algorithm:

1. Tokenize English, get `enc_out`.
2. Start with `[BOS]`.
3. Loop: run decoder on current prefix; take `argmax` at the last time step; stop at `[EOS]` or `max_target_length`.

**Note**: For better quality, replace with **beam search** (beam=4–6) and optional length penalty.

---

## 11) Shapes Cheat‑Sheet

* `input_ids`: `(B,S)`
* `attention_mask`: `(B,S)`
* `enc_out`: `(B,S,768)`
* `dec_inp`: `(B,T)`
* `labels`: `(B,T)`
* decoder hidden: `(B,T,768)`
* logits: `(B,T,|V|)`

---

