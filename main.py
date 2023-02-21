from seq_2_seq2 import Seq2SeqChatbot

PATHDATA = "data/AviationQA.csv"


def main():
    chatbot = Seq2SeqChatbot(PATHDATA)
    chatbot.load_data()
    (
        question_sequences,
        decoder_input_sequences,
        decoder_output_sequences,
    ) = chatbot.preprocess_data()
    chatbot.build_model(
        question_sequences, decoder_input_sequences, decoder_output_sequences
    )
    print(chatbot.generate_answer("What is your return policy?"))


if __name__ == "__main__":
    main()
