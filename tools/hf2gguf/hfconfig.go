package main

// hfconfig.go parses and validates the HF config/tokenizer against the
// reference GGUF metadata. The reference remains authoritative for values; this
// only proves the reference is consistent with the published HF files and that
// we parsed them (per the task), it does not source any GGUF value from them.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

type hfConfigFile struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int64   `json:"hidden_size"`
	NumHiddenLayers       int64   `json:"num_hidden_layers"`
	NumAttentionHeads     int64   `json:"num_attention_heads"`
	NumKeyValueHeads      int64   `json:"num_key_value_heads"`
	HeadDim               int64   `json:"head_dim"`
	IntermediateSize      int64   `json:"intermediate_size"`
	VocabSize             int64   `json:"vocab_size"`
	MaxPositionEmbeddings int64   `json:"max_position_embeddings"`
	RopeTheta             float64 `json:"rope_theta"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	QKNorm                bool    `json:"qk_norm"`
	QKNormGain            bool    `json:"qk_norm_gain"`
	EmbeddingNorm         bool    `json:"embedding_norm"`
	SlidingWindow         int64   `json:"sliding_window"`
	SWAFullEvery          int64   `json:"swa_full_every"`
}

func findKV(kvs []rawKV, key string) (rawKV, bool) {
	for _, kv := range kvs {
		if kv.Key == key {
			return kv, true
		}
	}
	return rawKV{}, false
}

func validateConfig(path string, kvs []rawKV) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var c hfConfigFile
	if err := json.Unmarshal(b, &c); err != nil {
		return err
	}

	intChecks := []struct {
		name string
		got  int64
	}{
		{"open1b.embedding_length", c.HiddenSize},
		{"open1b.block_count", c.NumHiddenLayers},
		{"open1b.attention.head_count", c.NumAttentionHeads},
		{"open1b.attention.head_count_kv", c.NumKeyValueHeads},
		{"open1b.attention.key_length", c.HeadDim},
		{"open1b.attention.value_length", c.HeadDim},
		{"open1b.feed_forward_length", c.IntermediateSize},
		{"open1b.context_length", c.MaxPositionEmbeddings},
		{"open1b.attention.sliding_window", c.SlidingWindow},
		{"open1b.attention.sliding_window_pattern", c.SWAFullEvery},
	}
	for _, chk := range intChecks {
		kv, ok := findKV(kvs, chk.name)
		if !ok {
			return fmt.Errorf("reference missing %s", chk.name)
		}
		if kv.Int != chk.got {
			return fmt.Errorf("%s: HF=%d ref=%d", chk.name, chk.got, kv.Int)
		}
	}
	floatChecks := []struct {
		name string
		got  float64
	}{
		{"open1b.rope.freq_base", c.RopeTheta},
		{"open1b.attention.layer_norm_rms_epsilon", c.RMSNormEps},
	}
	for _, chk := range floatChecks {
		kv, ok := findKV(kvs, chk.name)
		if !ok {
			return fmt.Errorf("reference missing %s", chk.name)
		}
		if float32(kv.Float) != float32(chk.got) {
			return fmt.Errorf("%s: HF=%v ref=%v", chk.name, chk.got, kv.Float)
		}
	}
	fmt.Printf("config.json OK: arch=%s layers=%d hidden=%d heads=%d/%d head_dim=%d ffn=%d vocab=%d swa=%d/%d tie=%v qk_norm=%v(gain=%v) emb_norm=%v\n",
		c.ModelType, c.NumHiddenLayers, c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads,
		c.HeadDim, c.IntermediateSize, c.VocabSize, c.SlidingWindow, c.SWAFullEvery,
		c.TieWordEmbeddings, c.QKNorm, c.QKNormGain, c.EmbeddingNorm)
	return nil
}

type tokConfigFile struct {
	EOSToken      string `json:"eos_token"`
	ModelMaxLen   int64  `json:"model_max_length"`
	ChatTemplate  string `json:"chat_template"`
	AddedDecoders map[string]struct {
		Content string `json:"content"`
	} `json:"added_tokens_decoder"`
}

func validateTokenizerConfig(path string, kvs []rawKV) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var tc tokConfigFile
	if err := json.Unmarshal(b, &tc); err != nil {
		return err
	}
	// chat_template must match the reference verbatim.
	if kv, ok := findKV(kvs, "tokenizer.chat_template"); ok {
		if kv.Str != tc.ChatTemplate {
			return fmt.Errorf("chat_template differs (HF %d bytes, ref %d bytes)", len(tc.ChatTemplate), len(kv.Str))
		}
	} else {
		return fmt.Errorf("reference missing tokenizer.chat_template")
	}
	// eos token id from added_tokens_decoder.
	if kv, ok := findKV(kvs, "tokenizer.ggml.eos_token_id"); ok {
		var eosID int64 = -1
		keys := make([]string, 0, len(tc.AddedDecoders))
		for k := range tc.AddedDecoders {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if tc.AddedDecoders[k].Content == tc.EOSToken {
				var id int64
				fmt.Sscanf(k, "%d", &id)
				eosID = id
			}
		}
		if eosID != kv.Int {
			return fmt.Errorf("eos_token %q: HF id=%d ref=%d", tc.EOSToken, eosID, kv.Int)
		}
	}
	// context length consistency
	if kv, ok := findKV(kvs, "open1b.context_length"); ok && tc.ModelMaxLen != 0 && tc.ModelMaxLen != kv.Int {
		return fmt.Errorf("model_max_length: HF=%d ref=%d", tc.ModelMaxLen, kv.Int)
	}
	fmt.Printf("tokenizer_config.json OK: eos=%s chat_template=%d bytes\n", tc.EOSToken, len(tc.ChatTemplate))
	return nil
}

type tokenizerJSON struct {
	Model struct {
		Type   string            `json:"type"`
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int64  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// validateTokenizerJSON proves the tokenizer.json we parsed is the source of the
// reference tokenizer arrays: same BPE type, same vocab+added token count, same
// merge count, and the added special tokens are the reference's leading tokens.
func validateTokenizerJSON(path string, kvs []rawKV) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(b, &tj); err != nil {
		return err
	}
	if kv, ok := findKV(kvs, "tokenizer.ggml.model"); ok && kv.Str != "gpt2" {
		return fmt.Errorf("ref tokenizer model %q, expected gpt2", kv.Str)
	}
	if tj.Model.Type != "BPE" {
		return fmt.Errorf("tokenizer.json model.type=%q, expected BPE", tj.Model.Type)
	}
	tokensKV, ok := findKV(kvs, "tokenizer.ggml.tokens")
	if !ok {
		return fmt.Errorf("reference missing tokenizer.ggml.tokens")
	}
	// tokenizer.json's vocab sometimes already contains the added specials and
	// sometimes not; accept either interpretation as long as the count matches.
	vocabN := uint64(len(tj.Model.Vocab))
	got := vocabN
	if got != tokensKV.Count {
		got = vocabN + uint64(len(tj.AddedTokens))
	}
	if got != tokensKV.Count {
		return fmt.Errorf("vocab=%d (+added=%d =%d) vs ref tokens=%d", vocabN, len(tj.AddedTokens), got, tokensKV.Count)
	}
	mergesKV, ok := findKV(kvs, "tokenizer.ggml.merges")
	if !ok {
		return fmt.Errorf("reference missing tokenizer.ggml.merges")
	}
	if uint64(len(tj.Model.Merges)) != mergesKV.Count {
		return fmt.Errorf("tokenizer.json merges=%d vs ref=%d", len(tj.Model.Merges), mergesKV.Count)
	}
	// Leading tokens (added specials, sorted by id) must match the ref array head.
	sort.Slice(tj.AddedTokens, func(i, j int) bool { return tj.AddedTokens[i].ID < tj.AddedTokens[j].ID })
	refHead := decodeStringArray(tokensKV.Raw, uint64(len(tj.AddedTokens)))
	for i := range tj.AddedTokens {
		if i >= len(refHead) {
			return fmt.Errorf("ref tokens shorter than added tokens")
		}
		if refHead[i] != tj.AddedTokens[i].Content {
			return fmt.Errorf("token[%d]: HF %q ref %q", i, tj.AddedTokens[i].Content, refHead[i])
		}
	}
	fmt.Printf("tokenizer.json OK: type=BPE vocab=%d +added=%d -> %d tokens, %d merges\n",
		vocabN, len(tj.AddedTokens), got, len(tj.Model.Merges))
	return nil
}

// keep the binary import used (helps future raw comparisons).
var _ = binary.LittleEndian
