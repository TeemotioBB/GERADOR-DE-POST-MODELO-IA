# Mídia + Vídeo Automático

Mini SaaS em Python para criar vídeos a partir de:

- uma **foto ou vídeo** novo para o primeiro take;
- um vídeo original contendo áudio e, opcionalmente, uma continuação;
- legenda automática ou texto fixo com suporte a **emojis**.

O sistema detecta onde o primeiro take termina, substitui esse trecho pela nova mídia e mantém o áudio do vídeo original do começo ao fim.

## Novidades desta versão

- O primeiro upload aceita imagem e vídeo.
- O tipo da mídia é reconhecido automaticamente.
- Vídeo inicial curto é repetido até a troca do take.
- Vídeo inicial longo é cortado exatamente na troca.
- O detector procura cortes mesmo quando o primeiro take original possui movimento.
- Textos manuais aceitam Unicode e emojis, com fontes Noto instaladas no Docker.

## Como funciona

1. Envie a nova mídia da personagem: foto ou vídeo.
2. Envie o vídeo original com o áudio.
3. O sistema detecta a primeira troca de take ou usa o segundo informado manualmente.
4. O trecho inicial é criado com a nova mídia.
5. A legenda é transcrita do áudio ou digitada manualmente.
6. A continuação original volta no ponto detectado.
7. O áudio do vídeo original é mantido.
8. A saída segue a proporção da primeira mídia.

## Estrutura

```text
.
├── app.py
├── core/
│   ├── __init__.py
│   ├── captions.py
│   ├── config.py
│   ├── media.py
│   ├── processor.py
│   └── transition.py
├── Dockerfile
├── railway.toml
├── requirements.txt
├── .env.example
├── run_local.bat
└── run_local.sh
```

> Importante: a pasta deve se chamar exatamente `core` e conter o arquivo `__init__.py`.

## Railway

Suba todos os arquivos para a raiz do repositório, mantendo a pasta `core`. O Railway usará o `Dockerfile` e a porta fornecida pela variável `PORT`.

Após o push no GitHub, o Railway fará um novo deploy automaticamente. O healthcheck é:

```text
GET /health
```

## Emojis

O Docker instala:

- DejaVu Sans;
- Noto Sans/Symbols;
- Noto Color Emoji.

As legendas são renderizadas como camadas PNG em Unicode. Isso evita os quadrados que o libass pode mostrar com alguns emojis e permite emojis coloridos quando a fonte do sistema oferece essa versão. Exemplos válidos:

```text
Quase de graça 😱🔥
Corre antes que acabe 🛍️✨
```

No Windows local, o FFmpeg também pode usar as fontes de emoji instaladas no sistema. Caso algum emoji muito novo apareça como quadrado, atualize o FFmpeg/fontes ou rode pelo Docker/Railway.

## Variáveis úteis

```env
WHISPER_MODEL=tiny
MAX_VIDEO_MINUTES=10
MAX_UPLOAD_SIZE=500mb
OUTPUT_CRF=20
OUTPUT_PRESET=veryfast
OUTPUT_MAX_LONG_EDGE=1920
FFMPEG_THREADS=1
BACKGROUND_BLUR_DIVISOR=4
TRANSITION_MAX_SCAN_SECONDS=90
```

`TRANSITION_MAX_SCAN_SECONDS` define até quantos segundos o detector procura a primeira troca. Para vídeos comuns de Reels/TikTok, 90 segundos é mais que suficiente.
