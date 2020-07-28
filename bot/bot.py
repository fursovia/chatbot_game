import logging
from aiogram import Bot, Dispatcher, executor, types
from chatgame.games.guess_topic_game import GuessTopicGame, example_of_using_GuessTopicGame
from bot.post_processing import *
from typing import Union

# guess_game_first_post_processing, guess_game_second_post_processing,CALLBACK_IN_GUESS_TOPIC_GAME

API_TOKEN = "1320827829:AAE3eWAgjBQ5HGrG80uUCTO-65-FtNXmKVY"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

GAMES = {"1": GuessTopicGame}
GAMES_NAMES = {'1': 'Guess topic'}

CALLBACK_SELECT_GAME = 'select_game_'

GAMES_FUNCTION_FIRST_POST_PROCESSING = {'1': guess_game_first_post_processing}
GAMES_FUNCTION_SECOND_POST_PROCESSING = {'1': guess_game_second_post_processing}

# словарь игровых сессий, где ключом является id пользователя (callback_query.from_user.id)
game_sessions = {}
# Код игры выбранный пользователем, где ключом является id пользователя (callback_query.from_user.id)
all_codes_of_game = {}
# Язык игры, ключом - id пользователя (callback_query.from_user.id)
language_of_games = {}

LANGUAGES = {'en': 'English',
             'ru': 'Russian'}

# False = чисто английский режим
# True  = возможность выбора генерации на английском/русском
IS_MULTILINGUAL_MODE = True


@dp.message_handler(commands=['start'])
async def greetings(message: types.Message):
    """
    This handler will be called when user sends `/start` or command
    """
    # TODO Добавлять сюда кнопки для новых игр

    choose_games_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                       one_time_keyboard=True,
                                                       reply=False)
    for game_id, game_name in GAMES_NAMES.items():
        button_game = types.InlineKeyboardButton(game_name,
                                                 callback_data=CALLBACK_SELECT_GAME + game_id)
        choose_games_keyboard.add(button_game)

    await message.reply("Hi!\nWhat game shall we play?",
                        reply_markup=choose_games_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda c: IS_MULTILINGUAL_MODE and c.data and c.data.startswith(CALLBACK_SELECT_GAME))
async def start_game_multilingual(callback_query: types.CallbackQuery):

    # Скрываем кнопки после выбора игры и пробрасываем в следующий хендлер где создается экземпляр класса игры
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    select_language_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                          one_time_keyboard=True,
                                                          reply=False)
    select_language_keyboard.add(
        types.InlineKeyboardButton('English', callback_data='lang_en' + callback_query.data),
        types.InlineKeyboardButton('Russian', callback_data='lang_ru' + callback_query.data)
    )

    await callback_query.message.reply("On what language do you want to play?",
                                       reply_markup=select_language_keyboard,
                                       reply=False)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith((CALLBACK_SELECT_GAME, 'lang_')))
async def start_game(callback_query: types.CallbackQuery):
    """
    В зависимости от нажатой кнопки в функции greetings здесь запускается
    одна из игр, определенных как машины состояний в словаре GAMES
    :param callback_query:
    :return:
    """

    # Считываем из callback код игры и язык, если мультиязычность включена (IS_MULTILINGUAL_MODE)
    language = 'en'
    if IS_MULTILINGUAL_MODE:
        language = callback_query.data[5:7]
        code_of_game = callback_query.data[len('lang_xx'+CALLBACK_SELECT_GAME):]
    else:
        code_of_game = callback_query.data[len(CALLBACK_SELECT_GAME):]

    # Сохраняем в глобальный словарь код игры
    all_codes_of_game[callback_query.from_user.id] = code_of_game

    # Сохраняем в глобальный словарь язык игры
    language_of_games[callback_query.from_user.id] = language

    # Скрываем кнопки после выбора игры
    await bot.delete_message(chat_id=callback_query.message.chat.id,
                             message_id=callback_query.message.message_id)

    await callback_query.message.reply("You choose the '{0}' game on {1} language!".format(GAMES_NAMES[code_of_game],
                                                                                           LANGUAGES[language]),
                                       reply=False)

    conditional_text_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                           one_time_keyboard=True,
                                                           reply=False)
    conditional_text_keyboard.add(
        types.InlineKeyboardButton('Yes', callback_data='conditional_1'),
        types.InlineKeyboardButton('No', callback_data='conditional_0')
    )

    await callback_query.message.reply("Would you like to write some initial text?",
                                       reply_markup=conditional_text_keyboard,
                                       reply=False)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('conditional_'))
async def conditional_text_or_not(callback_query: types.CallbackQuery):
    is_conditional = callback_query.data[len('conditional_'):]
    is_conditional = is_conditional[0]

    # Скрываем кнопки после выбора с затравкой или без
    await bot.delete_message(chat_id=callback_query.message.chat.id,
                             message_id=callback_query.message.message_id)

    if int(is_conditional):
        await callback_query.message.reply("Write out text:",
                                           reply=False)
    else:
        await in_generate_text('', callback_query.from_user.id, callback_query.message)


@dp.message_handler()
async def generate_text(message: types.Message):
    await in_generate_text(message.text, message.from_user.id, message)


async def in_generate_text(message_text, user_id, message: types.Message, language='en'):
    if message_text:
        text = message_text
    else:
        text = ''

    code_of_game = all_codes_of_game[user_id]
    game_class = GAMES[code_of_game]

    language = language_of_games[user_id]

    # Создаем экземпляр класса игры и сохраняем по user_id в словаря
    game_sessions[user_id] = game_class(language=language)
    game = game_sessions[user_id]

    game.triger_start_game(conditional_text_prefix=text)

    # Отправляем сгенерированный текст пользователю
    game.triger_receive_text()

    await message.reply(game.pert_gen_texts[0],
                        reply=False)

    # В зависимости от игры вызываем свою функцию
    post_processing_first = GAMES_FUNCTION_FIRST_POST_PROCESSING[code_of_game]

    await post_processing_first(game, message)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('first_post_processing_answer_'))
async def check_answer(callback_query: types.CallbackQuery):

    code_of_game = all_codes_of_game[callback_query.from_user.id]

    post_processing_second = GAMES_FUNCTION_SECOND_POST_PROCESSING[code_of_game]
    answer = await post_processing_second(bot, callback_query)

    game_inside = game_sessions[callback_query.from_user.id]
    game_inside.triger_finish_game(answer)

    await callback_query.message.reply(game_inside.final_message,
                                       reply=False)


if __name__ == '__main__':
    executor.start_polling(dp,
                           skip_updates=True)
