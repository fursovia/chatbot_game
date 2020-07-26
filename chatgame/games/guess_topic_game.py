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
    states = ['start', 'text_generation', 'text_received', 'users_turn']

    transitions = [
        {'trigger': 'triger_start_game',
         'source': 'start',
         'dest': 'text_generation',
         'after': 'start'},

        {'trigger': 'triger_receive_text',
         'source': 'text_generation',
         'dest': 'text_received',
         'after': 'receive_text'},

        {'trigger': 'triger_finish_game',
         'source': 'text_received',
         'dest': 'users_turn',
         'before': 'is_user_win'}
    ]

    final_message = ''

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
        """
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def start(self, *args, **kwargs):
        """
        Вызывается при 'triger_start_game'
        для перехода от состояния 'start'
        к состоянию 'text_generation',
        :return:
        """
        pass

    @abstractmethod
    def receive_text(self):
        """
        Вызывается при 'triger_receive_text',
        для перехода от состояния 'text_generation'
        к состоянию 'text_received'
        :return:
        """
        pass


class GuessTopicGame(AbstractGame):
    """
    Случайно выбирает тему и генерирует текст, а пользователь должен угадать эту тему.
    """

    def __init__(self):

        self.game = Machine(model=self,
                            states=GuessTopicGame.states,
                            initial='start',
                            transitions=GuessTopicGame.transitions)
                            # ignore_invalid_triggers = True)

        self.model_name = "gpt2-medium"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = initialize_model_and_tokenizer(model_name=self.model_name,
                                                                    device=self.device)
        self.bow = None or self.select_bow()
        self.discriminator = None or self.select_discriminator()
        self.model_hyperparameters = self.select_model_hyperparameters()

        self.topics = list(BAG_OF_WORDS_ADDRESSES.keys())
        self.random_chosen_topic = ''
        self.unpert_gen_text = ''
        self.pert_gen_texts = []

    def start(self, *, conditional_text_prefix=''):
        if self.topics:
            self.random_chosen_topic = random.choice(self.topics)

        self.context = prepare_text_primer(tokenizer=self.tokenizer,
                                           cond_text=conditional_text_prefix,
                                           device=self.device)

    def receive_text(self):
        pert_tok_texts = full_text_generation(model=self.model,
                                              tokenizer=self.tokenizer,
                                              context=self.context,
                                              device=self.device,
                                              bag_of_words=self.bow,
                                              **self.model_hyperparameters)
        self.pert_tok_texts = pert_tok_texts

        self.tokens_postprocessing()

    def tokens_postprocessing(self, *args, **kwargs):
        for i, pert_gen_tok_text in enumerate(self.pert_tok_texts):
            decoded_gen_text = decode_text_from_tokens(tokenizer=self.tokenizer,
                                                       tokens=pert_gen_tok_text)

            self.pert_gen_texts.append(decoded_gen_text)

    def is_user_win(self, topic_selected_by_user):
        print("\n\nin is_user_win {0}, {1}\n\n".format(topic_selected_by_user, self.random_chosen_topic))

        if topic_selected_by_user and topic_selected_by_user.lower() == self.random_chosen_topic:
            user_status = 'win'
        else:
            user_status = 'loose'

        self.final_message = 'You {0} the game!'.format(user_status)

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
        self.triger_start_game(conditional_text_prefix='The world ')
        self.triger_receive_text()
        self.triger_finish_game('military_test')


def example_of_using_GuessTopicGame(game_=None):
    game = GuessTopicGame()
    game.triger_start_game(conditional_text_prefix='The world ')
    game.triger_receive_text()
    game.triger_finish_game('military')

    game1 = GuessTopicGame()
    game1.triger_start_game(conditional_text_prefix='The forest ')
    game1.triger_receive_text()
    game1.triger_finish_game('science')

    game2 = GuessTopicGame()
    game2.triger_start_game(conditional_text_prefix='The forest ')
    game2.triger_receive_text()
    game2.triger_finish_game('science')


if __name__ == '__main__':
    example_of_using_GuessTopicGame()
