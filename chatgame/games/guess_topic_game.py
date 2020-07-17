from transitions import Machine
import random
import typing as tp

from chatgame.language_models.gpt2.gpt2 import run_pplm_example

class GuessTopicGame:
    """
    Случайно выбирает тему и генерирует текст, а пользователь должен угадать эту тему.
    """
    states = ['start', 'topics_received', 'topic_selected',
              'text_generated', 'text_received', 'user_win', 'user_loose']

    transitions = [
        {'trigger': 'start_game',
         'source': 'start',
         'dest': 'topics_received',
         'before': 'request_topics_from_clf'},

        {'trigger': 'select_topic',
         'source': 'topics_received',
         'dest': 'topic_selected',
         'before': 'choose_topic'},

        {'trigger': 'request_text',
         'source': 'topic_selected',
         'dest': 'text_generated',
         'before': 'request_text_from_clf'},

        {'trigger': 'receive_text',
         'source': 'text_generated',
         'dest': 'text_received',
         'before': 'receive_text_to_telegram'},

        {'trigger': 'user_choose_topic',
         'source': 'text_received',
         'dest': 'user_win',
         'conditions': 'is_user_win',
         'after': 'send_result_to_telegram'},

        {'trigger': 'user_choose_topic',
         'source': 'text_received',
         'dest': 'user_loose',
         'after': 'send_result_to_telegram'}
    ]

    def __init__(self, len_of_text=40):

        self.topics = []
        self.random_chosen_topic = ''
        self.len_of_text = len_of_text

        self.game = Machine(model=self,
                            states=GuessTopicGame.states,
                            initial='start',
                            transitions=GuessTopicGame.transitions)

    def request_topics_from_clf(self):
        self.topics = ['religion', 'politics']

    def choose_topic(self):
        if self.topics:
            self.random_chosen_topic = random.choice(self.topics)

    def request_text_from_clf(self):
        pass

    def receive_text_to_telegram(self):
        print("running pplm example")
        run_pplm_example()

    def receive_topic_from_user(self) -> str:
        pass
        # return 'religion'

    def send_result_to_telegram(self):
        pass

    def is_user_win(self) -> bool:
        topic_selected_by_user = self.receive_topic_from_user()
        if topic_selected_by_user:
            return topic_selected_by_user.lower() == self.random_chosen_topic
        else:
            return False


def example_of_using_GuessTopicGame(game_=None):
    game = game_ or GuessTopicGame()
    game.start_game()
    game.select_topic()
    game.request_text()
    game.receive_text()
    game.user_choose_topic()


if __name__ == '__main__':
    example_of_using_GuessTopicGame()
