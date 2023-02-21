import numpy as np
import tensorflow as tf
import logging


# TODO i need to make you answer my question
class Seq2SeqChatbot:
    def __init__(
            self,
            dataset_path,
            max_seq_length=50,
            embedding_dim=100,
            batch_size=64,
            epochs=100,
            latent_dim=256,
    ):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim

        self.dataset_path = dataset_path
        self.questions = []
        self.answers = []
        self.tokenizer = None
        self.encoder_model = None
        self.decoder_model = None
        self.START_TOKEN_IDX = None
        self.END_TOKEN_IDX = None
        self.model = None

    def load_data(self):

        with open(self.dataset_path, "r") as f:
            logging.info("Loading the data set")
            lines = f.read().split("\n")
            for line in lines:
                if line:
                    if line.count(",") == 2:
                        first_comma_index = line.index(",")
                        input_string = line[:first_comma_index] + line[first_comma_index + 1:]
                        question, answer = input_string.split(",")
                        self.questions.append(question)
                        self.answers.append(answer)

    def preprocess_data(self):
        # create model tokenizer
        logging.info("Creating tokonizer")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
        self.tokenizer.fit_on_texts(self.questions + self.answers)

        # add start and end tokens
        logging.info("add start and end tokens")
        num_words = len(self.tokenizer.word_index) + 1
        self.START_TOKEN_IDX = num_words
        self.END_TOKEN_IDX = num_words + 1
        self.tokenizer.word_index["<start>"] = self.START_TOKEN_IDX
        self.tokenizer.word_index["<end>"] = self.END_TOKEN_IDX
        self.tokenizer.index_word[self.START_TOKEN_IDX] = "<start>"
        self.tokenizer.index_word[self.END_TOKEN_IDX] = "<end>"

        # converting questions to sequences of integers
        logging.info("Convert questions to sequences of integers.")
        question_sequences = self.tokenizer.texts_to_sequences(self.questions)
        question_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            question_sequences, maxlen=self.max_seq_length, padding="post"
        )

        # converting answers to sequences of integers too
        # later no need to add dType its smart enough to know
        logging.info("Convert answers to sequences of integers")
        answer_sequences = self.tokenizer.texts_to_sequences(self.answers)
        answer_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            answer_sequences, maxlen=self.max_seq_length, padding="post"
        )

        # creating decoder input and target sequences
        logging.info("Create decoder input and target sequences")
        decoder_input_sequences = np.zeros_like(answer_sequences)
        decoder_input_sequences[:, :-1] = answer_sequences[:, 1:]
        decoder_output_sequences = np.zeros_like(answer_sequences)
        decoder_output_sequences[:, 1:] = answer_sequences[:, :-1]
        print(decoder_output_sequences)

        return question_sequences, decoder_input_sequences, decoder_output_sequences

    def build_model(
            self, question_sequences, decoder_input_sequences, decoder_output_sequences
    ):
        # define the encoder
        logging.info("Define the encoder")
        encoder_inputs = tf.keras.Input(shape=(self.max_seq_length,))
        x = tf.keras.layers.Embedding(len(self.tokenizer.word_index) + 1, self.embedding_dim)(
            encoder_inputs
        )
        encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
            self.latent_dim, return_state=True
        )(x)
        encoder_states = [state_h, state_c]

        # define the decoder
        logging.info("Define the decoder")
        decoder_inputs = tf.keras.Input(shape=(None,))
        x = tf.keras.layers.Embedding(len(self.tokenizer.word_index) + 1, self.embedding_dim)(
            decoder_inputs
        )
        decoder_lstm = tf.keras.layers.LSTM(
            self.latent_dim, return_sequences=True, return_state=True
        )
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
        decoder_outputs = tf.keras.layers.Dense(
            len(self.tokenizer.word_index) + 1, activation="softmax"
        )(decoder_outputs)

        # define the full model
        logging.info("Define the full model")
        self.model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # train the model
        # TODO i think i have mistake here i followed this article when it comes to the encoder and decoder
        # https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
        logging.info("Training the model ... ")
        self.model.compile(
            optimizer="rmsprop", loss="sparse_categorical_crossentropy"
        )
        self.model.fit(
            [question_sequences, decoder_input_sequences],
            decoder_output_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        self.model.save('my_chat_model')

        # define the encoder model that will be used for inference
        logging.info("Define the encoder model for inference")
        self.encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

        # define the decoder model that will be used for inference
        logging.info("Define the decoder model for inference")
        decoder_state_input_h = tf.keras.Input(shape=(self.latent_dim,))
        decoder_state_input_c = tf.keras.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            x, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = tf.keras.layers.Dense(
            len(self.tokenizer.word_index) + 1, activation="softmax"
        )(decoder_outputs)
        self.decoder_model = tf.keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

    def generate_answer(self, question):
        # Convert the question to a sequence of integers.
        logging.info("Convert the question to a sequence of integers ...")
        question_sequence = self.tokenizer.texts_to_sequences([question])
        question_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            question_sequence, maxlen=self.max_seq_length, padding="post"
        )

        # getting the encoder model
        states_value = self.encoder_model.predict(question_sequence)

        # keep going only the start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.START_TOKEN_IDX

        # generating the answer
        answer = ""
        for i in range(self.max_seq_length):

            decoder_outputs, state_h, state_c = self.decoder_model.predict([target_seq] + states_value)
            idx = np.argmax(decoder_outputs[0, 0, :])

            word = self.tokenizer.index_word[idx]
            if word == "<end>" or len(answer.split()) == self.max_seq_length:
                break

            if i > 0:
                answer += " "
            answer += word

            # setting the target sequence for the next iteration to be the predicted word
            target_seq[0, 0] = idx
            states_value = [state_h, state_c]

        return answer
