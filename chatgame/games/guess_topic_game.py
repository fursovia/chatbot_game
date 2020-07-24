from transitions import Machine
import random
import torch
from numpy import random as np_random
from abc import ABC, abstractmethod
from chatgame.language_models.gpt2.gpt2 import full_text_generation
from chatgame.language_models.gpt2.gpt2 import BAG_OF_WORDS_ADDRESSES
from chatgame.utils.misc import decode_text_from_tokens
from chatgame.utils.model_loader import initialize_model_and_tokenizer, prepare_text_primer


class AbstractGame(ABC):
    """
    Абстрактный класс игры, реализуемой через машину состояний
    """

    @abstractmethod
    def select_bow(self, *args, **kwargs):
        """
        Метод выбора мешка/мешков слов для конкретной игры
        :return:
        """
        pass

    @abstractmethod
    def select_discriminator(self, *args, **kwargs):
        """
        Метод выбора одного или нескольких дискриминаторов для конкретной игры
        :return:
        """
        pass

    @abstractmethod
    def select_model_hyperparameters(self, *args, **kwargs):
        """
        Метод, возвращающий словарь с гиперпараметрами для языковой модели
        :return:
        """
        pass

    @abstractmethod
    def tokens_postprocessing(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self):
        """
        Метод, запускающий игры
        :return:
        """
        pass


class GuessTopicGame(AbstractGame):
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
         'after': 'tokens_postprocessing'},

        {'trigger': 'user_choose_topic',
         'source': 'text_received',
         'dest': 'user_loose',
         'after': 'tokens_postprocessing'}
    ]

    def __init__(self):

        self.game = Machine(model=self,
                            states=GuessTopicGame.states,
                            initial='start',
                            transitions=GuessTopicGame.transitions)

        self.model_name = "gpt2-medium"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = initialize_model_and_tokenizer(model_name=self.model_name,
                                                                    device=self.device)
        self.context = prepare_text_primer(tokenizer=self.tokenizer,
                                           cond_text="",
                                           device=self.device)
        self.bow = None or self.select_bow()
        self.discriminator = None or self.select_discriminator()
        self.model_hyperparameters = self.select_model_hyperparameters()

        self.unpert_gen_text = ''
        self.pert_gen_texts = []

    def request_topics_from_clf(self):
        self.topics = list(BAG_OF_WORDS_ADDRESSES.keys())

    def choose_topic(self):
        if self.topics:
            self.random_chosen_topic = random.choice(self.topics)

    def request_text_from_clf(self):
        pass

    def receive_text_to_telegram(self):
        pert_tok_texts = full_text_generation(model=self.model,
                                              tokenizer=self.tokenizer,
                                              context=self.context,
                                              device=self.device,
                                              bag_of_words=self.bow,
                                              **self.model_hyperparameters)
        self.pert_tok_texts = pert_tok_texts

    def receive_topic_from_user(self) -> str:
        pass
        # return 'religion'

    def tokens_postprocessing(self, *args, **kwargs):
        for i, pert_gen_tok_text in enumerate(self.pert_tok_texts):
            decoded_gen_text = decode_text_from_tokens(tokenizer=self.tokenizer,
                                                       tokens=pert_gen_tok_text)

            self.pert_gen_texts.append(decoded_gen_text)

    def is_user_win(self) -> bool:
        topic_selected_by_user = self.receive_topic_from_user()
        if topic_selected_by_user:
            return topic_selected_by_user.lower() == self.random_chosen_topic
        else:
            return False

    def select_bow(self):
        topics = [t for t in BAG_OF_WORDS_ADDRESSES]
        if topics:
            return random.choice(topics)

    def select_discriminator(self):
        return

    @staticmethod
    def set_seed():
        seed = 16
        torch.manual_seed(seed)
        np_random.seed(seed)

    def select_model_hyperparameters(self):
        length = {"length": 70,
                  "num_samples": 1}

        sampling = {"sample": True,
                    "temperature": 1.0,
                    "top_k": 10,
                    "gm_scale": 0.9}

        gradient_descent = {"gamma": 1.5,
                            "kl_scale": 0.01,
                            "decay": False,
                            "window_length": 0,
                            "grad_length": 10000,
                            "num_iterations": 3,
                            "loss_type": 1,
                            "stepsize": 0.02}

        res = {}
        res.update(length)
        res.update(sampling)
        res.update(gradient_descent)
        return res

    def run(self):
        self.start_game()
        self.select_topic()
        self.request_text()
        self.receive_text()
        self.user_choose_topic()


def example_of_using_GuessTopicGame(game_=None):
    game = game_ or GuessTopicGame()
    game.start_game()
    game.select_topic()
    game.request_text()
    game.receive_text()
    game.user_choose_topic()


if __name__ == '__main__':
    example_of_using_GuessTopicGame()
