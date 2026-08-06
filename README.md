# Mídia + Vídeo Automático

Mini SaaS em Python que substitui o primeiro take de um vídeo por uma foto ou outro vídeo, recria a legenda e mantém o áudio e a continuação do original.

Nesta versão, o vídeo original pode ser fornecido de duas formas:

- anexando o arquivo normalmente;
- colando o link de um Reels público do Instagram, sem precisar baixá-lo antes.

## Como usar

1. Envie a nova mídia da personagem: foto ou vídeo.
2. No campo **Vídeo original com áudio e possível continuação**, escolha uma opção:
   - anexe o vídeo; ou
   - cole o link do Reels e clique em **IMPORTAR VÍDEO PELO LINK**.
3. Se desejar, clique em **Analisar troca do take**.
4. Escolha a forma de detectar a transição e configurar a legenda.
5. Clique em **GERAR VÍDEO**.

Depois da importação pelo link, o vídeo baixado aparece automaticamente no campo 2 e pode ser processado exatamente como um arquivo anexado.

## Recursos

- primeiro take com foto ou vídeo;
- repetição automática quando o vídeo inicial é curto;
- corte automático quando o vídeo inicial é longo;
- detecção automática ou manual da troca de take;
- manutenção do áudio original;
- continuação com fundo desfocado, barras pretas ou preenchimento da tela;
- transcrição com Whisper;
- leitura de texto gravado no vídeo por OCR;
- texto manual com Unicode e emojis;
- importação de Reels usando `yt-dlp`;
- nova tentativa com cookies quando o Instagram bloqueia o IP do Railway.

## Estrutura

```text
.
├── app.py
├── core/
│   ├── __init__.py
│   ├── captions.py
│   ├── config.py
│   ├── instagram_import.py
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

> A pasta deve se chamar exatamente `core` e conter o arquivo `__init__.py`.

## Railway

Suba todos os arquivos para a raiz do repositório, mantendo a pasta `core`. O Railway utilizará o `Dockerfile` e a porta definida em `PORT`.

O healthcheck é:

```text
GET /health
```

Após atualizar os arquivos no GitHub, o Railway fará o novo deploy automaticamente.

## Importação do Instagram e cookies

O sistema usa exatamente o mesmo módulo `instagram_import.py` do Gerador de Memes: tenta baixar o Reels sem autenticação primeiro e, quando o Instagram bloqueia o IP do Railway, tenta novamente com os cookies configurados no próprio serviço.

> **Importante:** as variáveis do Railway não são compartilhadas entre projetos ou serviços. Mesmo que `INSTAGRAM_COOKIES_B64` já exista no Gerador de Memes, você precisa copiar a mesma variável para este novo serviço.

Variáveis aceitas:

```env
MAX_INSTAGRAM_DOWNLOAD_MB=500
INSTAGRAM_COOKIES_B64=
INSTAGRAM_COOKIES=
INSTAGRAM_FORCE_IPV4=0
```

A opção recomendada é `INSTAGRAM_COOKIES_B64`:

1. Entre no Instagram pelo navegador.
2. Exporte os cookies do domínio `instagram.com` no formato Netscape `cookies.txt`.
3. Converta o arquivo para Base64.
4. No Railway, abra **Variables** e crie `INSTAGRAM_COOKIES_B64` com o conteúdo gerado.
5. Faça um novo deploy ou reinicie o serviço.

Linux/macOS:

```bash
base64 -i cookies.txt | tr -d '\n'
```

PowerShell:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("cookies.txt"))
```

Os cookies ficam somente no servidor. O aplicativo cria um arquivo temporário durante a tentativa de download e o apaga em seguida.

## Variáveis principais

```env
WHISPER_MODEL=tiny
MAX_VIDEO_MINUTES=10
MAX_UPLOAD_SIZE=500mb
MAX_INSTAGRAM_DOWNLOAD_MB=500
OUTPUT_CRF=20
OUTPUT_PRESET=veryfast
OUTPUT_MAX_LONG_EDGE=1920
FFMPEG_THREADS=1
BACKGROUND_BLUR_DIVISOR=4
TRANSITION_MAX_SCAN_SECONDS=90
TEMP_MAX_AGE_HOURS=3
```

`TRANSITION_MAX_SCAN_SECONDS` define por quanto tempo o detector procura a primeira troca. `TEMP_MAX_AGE_HOURS` controla quando os downloads e resultados temporários antigos são apagados.
