import logging

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


@dp.message_handler(commands=['start'])
async def greetings(message: types.Message):
    """
    This handler will be called when user sends `/start` or command
    """
    # TODO Сделать набор кнопки для каждой из игр

    choose_games_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                       one_time_keyboard=True)

    button_guess_topic_game = types.InlineKeyboardButton('Guess topic', callback_data='1')
    choose_games_keyboard.add(button_guess_topic_game)

    await message.reply("Hi!\nWhat game shall we play?", reply_markup=choose_games_keyboard)


@dp.callback_query_handler()
async def start_game(callback_query: types.CallbackQuery):
    """
    В зависимости от нажатой кнопки в функции greetings здесь запускается
    одна из игр, определенных как машины состояний в словаре GAMES
    :param callback_query:
    :return:
    """
    code_of_game = callback_query.data[-1]

    await bot.send_message(callback_query.from_user.id, 'You choose {} game!'.format(GAMES_NAMES[code_of_game]))
    # Скрываем кнопки после выбора игры
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    game_class = GAMES[code_of_game]
    game = game_class()
    example_of_using_GuessTopicGame(game)

    await bot.send_message(callback_query.message.chat.id, game.pert_gen_texts[0])

    # old style:
    # await bot.send_message(message.chat.id, message.text)

    # await message.answer(message.text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
