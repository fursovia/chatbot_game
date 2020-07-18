import logging
import random

from aiogram import Bot, Dispatcher, executor, types

from chatgame.games.guess_topic_game import GuessTopicGame, example_of_using_GuessTopicGame

API_TOKEN = "1320827829:AAE3eWAgjBQ5HGrG80uUCTO-65-FtNXmKVY"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

GAMES = {"1": GuessTopicGame}
GAMES_NAMES = {'1': 'Guess topic'}

CALLBACK_GAME = 'select_game_'
CALLBACK_TOPIC_IN_GUESS_TOPIC_GAME = 'answer_topic_'


@dp.message_handler(commands=['start'])
async def greetings(message: types.Message):
    """
    This handler will be called when user sends `/start` or command
    """
    # TODO Добавлять сюда кнопки для новых игр

    choose_games_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                       one_time_keyboard=True,
                                                       reply=False)
    button_guess_topic_game = types.InlineKeyboardButton('Guess topic',
                                                         callback_data=CALLBACK_GAME+'1')
    choose_games_keyboard.add(button_guess_topic_game)

    await message.reply("Hi!\nWhat game shall we play?",
                        reply_markup=choose_games_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith(CALLBACK_GAME))
async def start_game(callback_query: types.CallbackQuery):
    """
    В зависимости от нажатой кнопки в функции greetings здесь запускается
    одна из игр, определенных как машины состояний в словаре GAMES
    :param callback_query:
    :return:
    """
    code_of_game = callback_query.data.strip(CALLBACK_GAME)

    # Скрываем кнопки после выбора игры
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    await callback_query.message.reply("You choose the '{}' game!".format(GAMES_NAMES[code_of_game]),
                                       reply=False)

    game_class = GAMES[code_of_game]
    game = game_class()
    example_of_using_GuessTopicGame(game)

    await callback_query.message.reply(game.pert_gen_texts[0],
                                       reply=False)

    # Кнопки с выбором вариантов тем.
    topics_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                 one_time_keyboard=True)

    # Для вывода юзеру игры GuessTopicGame выбираем 4 варианта тем или меньше, если количество тем меньше 4.
    # Причем обязательно добавляем тему, выбранную в движке игры для генерации текста.
    topic_variants = random.sample(set(game.topics) - set([game.random_chosen_topic]),
                                   min(3, len(game.topics)-1))
    topic_variants.append(game.random_chosen_topic)

    # Перемешиваем названия кнопок перед выводом на экран.
    random.shuffle(topic_variants)

    for topic in topic_variants:
        button_topic = types.InlineKeyboardButton(topic,
                                                  callback_data=CALLBACK_TOPIC_IN_GUESS_TOPIC_GAME+topic)
        topics_keyboard.add(button_topic)

    await callback_query.message.reply("Try to guess the topic of the previous text.",
                                       reply_markup=topics_keyboard,
                                       reply=False)
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    # await message.answer(message.text)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith(CALLBACK_TOPIC_IN_GUESS_TOPIC_GAME))
async def check_answer_in_guess_topic(callback_query: types.CallbackQuery):

    answer_topic = callback_query.data.strip(CALLBACK_TOPIC_IN_GUESS_TOPIC_GAME)

    await callback_query.message.reply("You select the '{}' topic!".format(answer_topic),
                                       reply=False)

    # Скрываем кнопки после выбора темы
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    # TODO: Передать истинный ответ 'game.random_chosen_topic' из хендлера 'start_game' и сравнить с 'answer_topic'

if __name__ == '__main__':
    executor.start_polling(dp,
                           skip_updates=True)
