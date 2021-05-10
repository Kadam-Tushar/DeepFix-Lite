import pandas as pd, tempfile
from absl import app
from absl import flags
import tokenization

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csv_file', 'train.csv', 'train.csv')

flags.DEFINE_string('output_csv_file', 'train_out.csv', 'train_out.csv')

flags.DEFINE_string('buggy_text', 'sourceText', 'Tag to identify buggy programs in the CSV file.')

flags.DEFINE_string('correct_text', 'targetText', 'Tag to identify correct programs in the CSV file.')

flags.DEFINE_string('buggy_text_tokens', 'sourceTokens', 'Tag to identify tokenized buggy programs in the CSV file.')

flags.DEFINE_string('correct_text_tokens', 'targetTokens', 'Tag to identify tokenized correct programs in the CSV file.')

flags.DEFINE_string('buggy_line', 'sourceLineText', 'Tag to identify buggy lines in the CSV file.')

flags.DEFINE_string('correct_line', 'targetLineText', 'Tag to identify correct lines in the CSV file.')

flags.DEFINE_string('buggy_line_tokens', 'sourceLineTokens', 'Tag to identify tokenized buggy lines in the CSV file.')

flags.DEFINE_string('correct_line_tokens', 'targetLineTokens', 'Tag to identify tokenized correct lines in the CSV file.')


def main(argv):
    df = pd.read_csv(FLAGS.input_csv_file)
    tokenizer = tokenization.C_Tokenizer()
    
    buggy_text_tokens_list = []
    correct_text_tokens_list = []
    buggy_line_tokens_list = []
    correct_line_tokens_list = []
    
    for _, row in df.iterrows():
        buggy_line = row[FLAGS.buggy_line]
        buggy_line_tokens, _ = tokenizer.tokenize(buggy_line)
        
        correct_line = row[FLAGS.correct_line]
        correct_line_tokens, _ = tokenizer.tokenize(correct_line)

        assert buggy_line_tokens != correct_line_tokens

        buggy_line_tokens_list.append(buggy_line_tokens)
        correct_line_tokens_list.append(correct_line_tokens)

        buggy_text = row[FLAGS.buggy_text]
        correct_text = row[FLAGS.correct_text]
        
        buggy_text_tokens = []
        for line in buggy_text.splitlines():
            tokens, _ = tokenizer.tokenize(line)
            buggy_text_tokens.append(tokens)
        buggy_text_tokens_list.append(buggy_text_tokens)

        correct_text_tokens = []
        for line in correct_text.splitlines():
            tokens, _ = tokenizer.tokenize(line)
            correct_text_tokens.append(tokens)
        correct_text_tokens_list.append(correct_text_tokens)

        
    df[FLAGS.buggy_text_tokens] = buggy_text_tokens_list
    df[FLAGS.correct_text_tokens] = correct_text_tokens_list
    df[FLAGS.buggy_line_tokens] = buggy_line_tokens_list
    df[FLAGS.correct_line_tokens] = correct_line_tokens_list

    df.to_csv(FLAGS.output_csv_file, index=False)


if __name__ == '__main__':
    app.run(main)
