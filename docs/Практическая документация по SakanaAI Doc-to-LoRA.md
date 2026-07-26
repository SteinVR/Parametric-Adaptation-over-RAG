# Практическая документация по SakanaAI Doc-to-LoRA

## Обзор

Doc-to-LoRA (D2L) — это референс-имплементация гиперсети, которая по входному документу генерирует LoRA‑адаптер для базовой LLM, чтобы модель «внутренне запомнила» факты без классического fine-tuning или длинного контекста. Код опубликован SakanaAI как открытый проект, с предобученными чекпоинтами на Hugging Face и сопроводительной статьёй и веб‑демо. Для практического использования в коде достаточно поднять окружение, скачать готовый чекпоинт гиперсети и обернуть ModulatedPretrainedModel в ваш сервис.

## Установка окружения

Репозиторий рассчитан на Python не ниже 3.10, что зафиксировано в `pyproject.toml` проекта (`requires-python = ">= 3.10"`). Базовый и рекомендуемый способ установки — через менеджер окружений `uv`:

```bash
git clone https://github.com/SakanaAI/doc-to-lora.git
cd doc-to-lora
curl -LsSf https://astral.sh/uv/install.sh | sh
./install.sh
```


Скрипт `install.sh` внутри репозитория создаёт `uv`‑виртуальное окружение на Python 3.10, ставит фиксированную версию PyTorch 2.6.0 (CUDA 12.4) и связанные пакеты (`torchvision`, `torchaudio`), синхронизирует зависимости из `pyproject.toml`, а также отдельно доустанавливает совместимые версии `tokenizers`, `flash-attn` и `flashinfer` для ускоренного внимания. В конце он скачивает датасет SQuAD и прогоняет несколько `data/build_*.py` скриптов для подготовки компактных датасетов, что нужно только для воспроизведения обучения и экспериментов, но не обязательно для чистого инференса по предобученным моделям.

Если вы не хотите использовать `uv`, можно установить пакет как обычный Python‑модуль `ctx-to-lora`, так как именно так он объявлен в `pyproject.toml`, выполнив `pip install .` или `pip install -e .` из корня репозитория в заранее подготовленном окружении с подходящей версией PyTorch.

**Платформенные ограничения `install.sh`.** Предобученный wheel для `flash-attn` в `install.sh` собран строго под `cp310-cp310-linux_x86_64` (Python 3.10, Linux x86-64, CUDA 12, PyTorch 2.6, `cxx11abi=FALSE`). Аналогично, `flashinfer-python==0.2.2` ставится из индекса `https://flashinfer.ai/whl/cu124/torch2.6`. Если ваша платформа отличается (другая версия Python, macOS/Windows, другая сборка CUDA), `install.sh` не выполнится — в этом случае используйте ручную установку, описанную ниже.

## Установка в существующее окружение (без install.sh)

Если у вас уже есть рабочее Python-окружение (например, `.venv` проекта с подходящей версией PyTorch), Doc-to-LoRA можно установить туда, не запуская `install.sh` и не создавая отдельный venv.

### Минимальные требования

| Требование | Значение |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.6.0 с CUDA (протестировано на cu124) |
| GPU VRAM | ~16 GB для `gemma-2-2b-it` + гиперсеть в bfloat16; для моделей крупнее — пропорционально больше |
| ОС | Linux x86-64 (flash-attn wheel предсобран только под неё; без flash-attn работает через SDPA fallback) |

### Шаги

1. **Клонировать репозиторий** (если ещё не сделано):

```bash
git clone https://github.com/SakanaAI/doc-to-lora.git
```

2. **Установить пакет в текущее окружение** — из корня клонированного репо:

```bash
pip install -e ./doc-to-lora
```

Эта команда поставит все зависимости из `pyproject.toml`. Ключевые пакеты и их версии (зафиксированы авторами):

- `transformers==4.51.3`
- `accelerate==1.6.0`
- `peft` (без верхней границы)
- `deepspeed==0.17.1`
- `datasets==3.6.0`
- `vllm==0.8.5.post1`
- `bitsandbytes>=0.46.1`
- `gradio>=4.40.0`
- `einops`, `jaxtyping`, `liger-kernel`

Если какие-то из этих зависимостей конфликтуют с вашим проектом, можно изолировать Doc-to-LoRA в отдельный venv и обращаться к нему как к сервису (см. раздел "Паттерны интеграции").

3. **Flash Attention (опционально, но рекомендуется)**. Без flash-attn модель работает, но медленнее — через `sdpa` или `eager` fallback. Для установки:

```bash
pip install flash-attn --no-build-isolation
```

Либо точный wheel, как в `install.sh` (только Python 3.10 + CUDA 12 + PyTorch 2.6):

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Для `flashinfer` (ускоряет vLLM, но не обязателен для чистого инференса через `ModulatedPretrainedModel`):

```bash
pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6
```

4. **Скачать чекпоинты** — одинаково независимо от способа установки:

```bash
huggingface-cli login
huggingface-cli download SakanaAI/doc-to-lora \
  --local-dir trained_d2l --include "*/"
```

5. **Проверить установку**:

```python
import torch
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_flash_attn=True, use_sequence_packing=False
)
model = model.to("cuda").to(torch.bfloat16)
model.eval()
print("OK:", model.base_model.config.name_or_path)
```

Если `flash-attn` не установлен, передайте `use_flash_attn=False` — модель переключится на `eager` attention.

### Обязательное примечание: `.eval()`, dtype и device

В отличие от минимального примера из README, для корректного продакшен-инференса важно после загрузки модели:

- перевести её на GPU: `.to("cuda")` или `.to(device)`;
- привести к `bfloat16`: `.to(torch.bfloat16)` — экономит VRAM вдвое без потери качества;
- включить eval-режим: `.eval()` — отключает dropout и прочие training-only слои.

Именно так делает Gradio-демо в `demo/app.py`.

## Получение предобученных Doc‑to‑LoRA моделей

Авторы публикуют предобученные гиперсети Doc-to-LoRA в репозитории `SakanaAI/doc-to-lora` на Hugging Face Hub. Рекомендуемый в README способ скачивания чекпоинтов — через `huggingface-cli` под управлением `uv`:

```bash
uv run huggingface-cli login
uv run huggingface-cli download SakanaAI/doc-to-lora \
  --local-dir trained_d2l --include "*/"
```


Эта команда кладёт структуру чекпоинтов в каталог `trained_d2l`, откуда они далее подхватываются примерами кода и демо, например путь `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin` в официальном Python‑сниппете. В продакшене обычно стоит скачать нужные чекпоинты заранее (через тот же `huggingface-cli download` на стадии сборки образа или деплоя) и гарантировать, что путь к `pytorch_model.bin` доступен для чтения процессу инференса.

## Базовый Python API: однодокументный сценарий

README даёт минимальный пример работы с Doc-to-LoRA через высокоуровневый класс `ModulatedPretrainedModel`, обёртывающий базовую LLM с LoRA‑адаптером и гиперсетью. Этот интерфейс поддерживает только небатчевые входы, то есть один документ и один запрос за раз:

```python
import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# загрузка гиперсети из чекпоинта
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()

tokenizer = get_tokenizer(model.base_model.name_or_path)

# документ, который нужно "внутренизировать"
doc = open("data/sakana_wiki.txt", "r").read()
chat = [{"role": "user", "content": "Tell me about Sakana AI."}]

chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# один вызов internalize превращает документ в LoRA‑адаптер
model.internalize(doc)

# generate теперь использует внутренние LoRA‑веса вместо длинного контекста
outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```


Метод `internalize` внутри `ModulatedPretrainedModel` сам токенизирует строку документа, прогоняет её через контекстный энкодер, вызывает гиперсеть для генерации LoRA‑весов и кэширует их в поле `generated_loras`, не трогая базовую модель до фактической генерации. При последующем вызове `generate` без явного `ctx_ids` класс обнаруживает наличие ранее сгенерированных LoRA, объединяет их (при необходимости по чанкам) и патчит forward соответствующих PEFT‑слоёв базовой модели перед тем, как вызвать стандартный `.generate` у `base_model`.

Для сброса влияния документа предусмотрен метод `reset`, который восстанавливает исходный `forward` в целевых слоях и обнуляет кэш `generated_loras`, возвращая модель к поведению "как без Doc-to-LoRA" до следующего `internalize`.

## Расширенный API и батчинг

Файл `src/ctx_to_lora/modeling/hypernet.py` реализует полный класс `ModulatedPretrainedModel` с более низкоуровневым методом `generate`, который принимает не только токены чата, но и подготовленные тензоры контекста, а также дополнительные параметры, полезные для продакшена. Сигнатура включает аргументы `ctx_ids`, `ctx_attn_mask`, `ctx_position_ids`, `n_ctx_chunks`, `n_queries`, `scalers` и `bias_scaler`, после которых следуют стандартные `input_ids` и параметры генерации базовой модели.

Внутри `generate` реализованы два режима: если `ctx_ids` не передан и ранее был вызван `internalize`, будут использованы закэшированные LoRA‑веса в `self.generated_loras`; если же `ctx_ids` и `ctx_attn_mask` заданы явно, гиперсеть заново сгенерирует LoRA по переданному контексту. В обоих случаях затем вызывается функция `combine_lora`, которая умеет агрегировать несколько чанков контекста (по `n_ctx_chunks`) и применить скейлеры `scalers` и `bias_scaler` к весам и биасам адаптера, после чего вспомогательная функция `apply_lora_to_layers` патчит указанные слои базовой модели перед генерацией текста.

Для батчевого инференса можно напрямую вызывать `generate` с батчем `ctx_ids` (например, несколько документов или чанков), соответствующими масками и вектором `scalers` той же длины; в официальном Gradio‑демо показан пример, где `ctx_ids` формируется из одного контекста, но как батч размера 1 через паддинг, а `scalers` — это тензор `shape=(1,)` с выбранным на слайдере коэффициентом.

## Как устроена загрузка моделей и токенизаторов

Вспомогательный модуль `ctx_to_lora.model_loading` содержит функции `get_model`, `get_tokenizer` и `get_model_and_tokenizer`, которые используются как при обучении, так и в демо. `get_tokenizer` создаёт `AutoTokenizer` для заданного `model_name_or_path`, включает `trust_remote_code=True`, настраивает `padding_side` (слева для инференса) и `truncation_side="left"`, а при наличии файла `chat_templates/{model_name}.jinja` подхватывает jinja‑шаблон чата в поле `tokenizer.chat_template`.

Функция `get_model` создает или `AutoModelForCausalLM`, или `AutoModel` (для би‑дирекционных моделей типа BERT/GTE), или языковую часть `Gemma3ForConditionalGeneration` для vision‑моделей, настраивая реализацию внимания (`flash_attention_2` или `sdpa`), типы данных и опциональный 4‑битный квантованный режим через `BitsAndBytesConfig`; при наличии `peft_config` базовая модель оборачивается в `PeftModel`, а флаг `requires_grad` контролирует, остаются ли параметры обучаемыми. В `ModulatedPretrainedModel.from_state_dict` эта функция вызывается для восстановления ровно той же базовой модели, которая использовалась при мета‑обучении гиперсети, причём имя модели и конфигурация LoRA читаются из сохранённого `state_dict["hypernet_config"]` и `state_dict["base_model_name_or_path"]`.

## Паттерны интеграции в продакшен‑сервис

### Долгоживущий сервис с заранее загруженным Doc-to-LoRA

Наиболее естественный для продакшена паттерн повторяет структуру Gradio‑демо `demo/app.py`, где глобальные переменные хранят загруженную `ModulatedPretrainedModel`, токенизаторы и историю чата. При выборе чекпоинта из списка обнаруженных файлов `pytorch_model.bin` вызывается `ModulatedPretrainedModel.from_state_dict(..., train=False, use_flash_attn=True, use_sequence_packing=False)`, модель переводится на `device` (GPU при наличии) и в `bfloat16`, а энкодер контекста и основной токенизатор инициализируются через `get_tokenizer` для имён моделей, зашитых в `ctx_encoder_args` и `base_model.config.name_or_path`.

Далее при каждом запросе интерфейс делает следующее: подготавливает токены контекста функцией `tokenize_ctx_text` из `ctx_to_lora.data.processing`, паддит их в тензоры `ctx_ids` и `ctx_attn_mask`, а текст запроса и историю чата оборачивает через `base_tokenizer.apply_chat_template(..., add_generation_prompt=True)`; затем вызывает `modulated_model.generate`, передавая тензоры контекста, скаляр `context_scaler` как одномерный тензор `scalers_tensor` и скаляр `bias_scaler`, вместе с `input_ids` и параметрами генерации (`max_new_tokens`, `do_sample`, `temperature`). Результат декодируется тем же `base_tokenizer.decode`, при этом из сгенерированных токенов отсекается длина исходного промпта (`outputs[0][model_inputs.shape[1]:]`), чтобы вернуть только ответ модели.

Функция `tokenize_ctx_text` принимает словарь `{"context": [str, ...]}` и токенизатор контекстного энкодера, разбивает длинные документы на чанки по максимальной длине токенизатора и возвращает словарь с ключом `ctx_ids` — списком тензоров (по чанку на элемент). В демо эти чанки затем паддятся через `torch.nn.utils.rnn.pad_sequence` в батч и передаются в `generate`.

Аналогичным образом вместо Gradio можно построить HTTP‑сервис (например, на FastAPI или Flask), который при старте процесса один раз загружает Doc-to-LoRA чекпоинт и токенизаторы, а затем в каждом запросе реализует тот же цикл: подготовка `ctx_ids` и `input_ids`, вызов `generate`, декодирование и возврат ответа. Этот подход хорошо укладывается в архитектуру "отдельный сервис генерации адаптера/ответа" для более крупной системы, где upstream‑сервисы обеспечивают поставку документов и управление версиями LoRA.

### Режим "internalize один раз, много запросов"

Помимо схемы "контекст в каждом запросе", класс `ModulatedPretrainedModel` поддерживает сценарий, где документ или набор документов один раз "внутренизируется" через `internalize`, после чего серия запросов обслуживается без повторной генерации LoRA. В этом режиме можно, например, в обработчике события "документ обновился" заново вызвать `model.internalize(new_doc)` и затем в обычных эндпоинтах чата вызывать `model.generate(input_ids=...)` без параметров `ctx_ids`, полагаясь на закэшированный в `generated_loras` адаптер.

Для полного сброса знаний о документе сервис может по сигналу (например, при закрытии сессии) звать `model.reset()`, чтобы вернуть базовую модель к исходному состоянию без LoRA‑патчинга, что важно, если один и тот же процесс обслуживает разные не пересекающиеся по данным сессии или арендаторов.

## Настройка силы влияния контекста

В Gradio‑демо реализованы два управляемых пользователем параметра, отражающие возможности низкоуровневой функции `combine_lora`: слайдер `Context Scaling`, который управляет значением `context_scaler`, и слайдер `Bias Scaler`, который управляет параметром `bias_scaler`. В коде запроса этот скейлер контекста преобразуется в тензор `scalers_tensor` длины 1 и передаётся в `modulated_model.generate` как `scalers=scalers_tensor`, после чего влияет на масштабирование сгенерированных LoRA‑весов по сравнению с базовой моделью.

Согласно подписи в интерфейсе, `Bias Scaler` — это "single scalar applied to bias parameters (independent of contexts)", то есть единый коэффициент, умножающий сгенерированные биасы адаптера независимо от конкретного контекста. В прод‑сценариях эти параметры можно вынести в конфигурацию сервиса или управлять ими на уровне сессий, чтобы, например, ослаблять или усиливать влияние внутримодельной памяти относительно явного контекста или инструментов.

## Ограничения и оговорки

В шапке README и в демонстрационном приложении подчёркивается, что данный код является референсной исследовательской реализацией и используется для воспроизведения результатов статьи Doc-to-LoRA, а не как готовый коммерческий продукт. В самом веб‑интерфейсе есть явное предупреждение, что модель может галлюцинировать и не предназначена для использования в миссион‑критичных сценариях, а вся ответственность за её эксплуатацию возлагается на пользователя.

Также README отдельно отмечает, что приведённый выше "Python API Usage" работает только с небатчевыми входами, а для батчевого инференса и более сложных сценариев следует обращаться к низкоуровневым интерфейсам в `src/ctx_to_lora/modeling/hypernet.py`, что важно учитывать при проектировании высоконагруженного сервиса. В статье и официальном блоге Sakana AI подчёркивается, что сам подход Doc-to-LoRA обеспечивает существенную экономию VRAM за счёт переноса знания из длинного контекста в компактный LoRA‑адаптер, но конкретные характеристики времени отклика и потребления памяти зависят от выбранной базовой модели и железа.

## Проверка актуальности и соответствия коду

Данная документация основана на актуальном README репозитория (`README.md` на ветке `main`), файлах `pyproject.toml` и `install.sh`, а также на исходном коде модулей `ctx_to_lora/modeling/hypernet.py`, `ctx_to_lora/model_loading.py` и демонстрационном приложении `demo/app.py` из того же коммита. Все описанные функции (`ModulatedPretrainedModel.from_state_dict`, `internalize`, `generate`, `reset`, `get_model`, `get_tokenizer`, обработка контекста в демо) и их сигнатуры напрямую сверены с этими файлами и не опираются на несуществующие или устаревшие API.

Кроме того, сведения о доступности кода и предобученных моделей, а также о назначении метода Doc-to-LoRA, дополнительно проверены по официальной странице Sakana AI о Doc-to-LoRA/Text-to-LoRA и связанным материалам. При обновлении репозитория (например, смене версий зависимостей или добавлении новых демо) стоит ещё раз свериться с README и соответствующими модулями, но на момент марта 2026 года изложённые шаги по установке и использованию соответствуют состоянию открытого кода.
