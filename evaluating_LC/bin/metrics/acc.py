

class Accuracy:
    def __init__(self, format: str, exclude_not_attempted: bool):
        self.format = format
        self.exclude_not_attempted = exclude_not_attempted

    def evaluate(self, responses_df):
        if self.format == "simpleqa_like":
            correct_count = (responses_df["accuracies"] == "CORRECT").sum()
            incorrect_count = (responses_df["accuracies"] == "INCORRECT").sum()
            not_attempted_count = (responses_df["accuracies"] == "NOT_ATTEMPTED").sum()
            if self.exclude_not_attempted:
                accuracy = correct_count / (correct_count + incorrect_count)
            else:
                accuracy = correct_count / (correct_count + incorrect_count + not_attempted_count)
            return accuracy
        else:
            raise ValueError(f"Invalid format: {self.format}")