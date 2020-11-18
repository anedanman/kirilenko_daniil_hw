# Запуск
`docker-compose up`

# Модель ответов на вопросы по изображениям:
В качестве модели использовалась простейшая модель iBOWIMG (https://arxiv.org/pdf/1512.02167.pdf):
Эмбединги вопроса получаются с помощью Bag-of-Words, эмбединги изображения с помощью ResNet18,
все они конкатенируются и подаются на вход классификатору по пространству ответов.