from abc import ABC
from torch import nn
from transformers import VisionEncoderDecoderModel, ViTConfig, BertConfig, VisionEncoderDecoderConfig


#an abstract model used as a base for other models
class Model(ABC): 
    def __init__(self):
        super().__init__()

#a model for fine-tuning transformers. Used for GET medium and large

class FineTunedTransformerGEGT(nn.Module):
    def __init__(self, vocab_size):
        super(FineTunedTransformerGEGT, self).__init__()

        # Initialize the vision encoder and text decoder configuration
        config_encoder = ViTConfig()
        config_decoder = BertConfig()

        # Combine them into a VisionEncoderDecoderConfig
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        # Load the VisionEncoderDecoderModel with the specified config
        self.model = VisionEncoderDecoderModel(config=config)

        # Define a final linear layer to project to the output vocab size
        self.lm_head = nn.Linear(self.model.config.decoder.hidden_size, vocab_size)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        # Get the encoder outputs (vision transformer features)
        encoder_outputs = self.model.encoder(pixel_values)

        # Pass the encoder outputs to the decoder
        decoder_outputs = self.model.decoder(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             encoder_hidden_states=encoder_outputs.last_hidden_state,
                                             encoder_attention_mask=None,
                                             output_hidden_states=True)  # Ensure hidden states are returned

        # The decoder outputs are a tuple, where the first element is the logits and the second is the hidden states
        decoder_hidden_states = decoder_outputs.hidden_states  # Get all hidden states (if needed)
        decoder_last_hidden_state = decoder_hidden_states[-1] # The last hidden state

        # Use the decoder's last hidden state to compute logits
        logits = self.lm_head(decoder_last_hidden_state)

        # If labels are provided, compute the loss
        if labels is not None:
            # Shift the labels for causal language modeling (if necessary)
            shift_logits = logits[:, :-1, :].contiguous()  # Remove last token from logits
            shift_labels = labels[:, 1:].contiguous()  # Remove first token from labels
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss  # Return the loss for training

        return logits  # Return logits if labels are not provided for inference