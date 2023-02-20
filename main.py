# This is a sample Python script.
from seq_2_seq2 import Seq2SeqChatbot

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
PATHDATA = "data/AviationQA.csv"


def play():
    # Use a breakpoint in the code line below to debug your script.
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


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    play()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
