from aiogram import types
import random


async def guess_game_first_post_processing(game, message: types.Message):

    # Кнопки с выбором вариантов тем.
    topics_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                 one_time_keyboard=True)

    # Для вывода юзеру игры GuessTopicGame выбираем 4 варианта тем или меньше, если количество тем меньше 4.
    # Причем обязательно добавляем тему, выбранную в движке игры для генерации текста.
    topic_variants = random.sample(set(game.topics) - set([game.bow]),
                                   min(3, len(game.topics) - 1))
    topic_variants.append(game.bow)

    # Перемешиваем названия кнопок перед выводом на экран.
    random.shuffle(topic_variants)

    for topic in topic_variants:
        # Через callback передаем истинную тему текста (после знака ':')
        button_topic = types.InlineKeyboardButton(topic,
                                                  callback_data='first_post_processing_answer_' + topic +
                                                                ':' + game.bow)
        topics_keyboard.add(button_topic)

    await message.reply("Try to guess the topic of the previous text.",
                        reply_markup=topics_keyboard,
                        reply=False)


async def guess_game_second_post_processing(bot, callback_query: types.CallbackQuery):
    # из callback берем выбранную пользователем тему, а также истинную тему
    answer_topic = callback_query.data[len('first_post_processing_answer_'):]
    ind_true_answer = answer_topic.find(':')
    true_topic = answer_topic[ind_true_answer + 1:]
    answer_topic = answer_topic[:ind_true_answer]

    await callback_query.message.reply("You select the '{0}' topic!".format(answer_topic),
                                       reply=False)

    # Скрываем кнопки после выбора темы
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    await callback_query.message.reply("True topic is '{0}'.".format(true_topic),
                                       reply=False)

    return answer_topic
