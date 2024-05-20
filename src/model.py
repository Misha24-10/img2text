from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from src.config import image_encoder_model, text_decode_model, output_dir, ModelUpdate

feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
text_tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)

if text_decode_model in ["GPT2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    text_tokenizer.pad_token = text_tokenizer.eos_token

model.config.update({
    "vocab_size": model.config.decoder.vocab_size,
    "eos_token_id": text_tokenizer.eos_token_id,
    "decoder_start_token_id": text_tokenizer.bos_token_id,
    "pad_token_id": text_tokenizer.pad_token_id,
    "max_length": ModelUpdate.max_length,
    "early_stopping": ModelUpdate.early_stopping,
    "no_repeat_ngram_size": ModelUpdate.no_repeat_ngram_size,
    "length_penalty": ModelUpdate.length_penalty,
    "num_beams": ModelUpdate.num_beams
})

def save_model():
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    text_tokenizer.save_pretrained(output_dir)