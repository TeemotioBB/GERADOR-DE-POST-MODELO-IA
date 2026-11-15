# Studio Reels 2.0 — Gerador de vídeo com nova interface e uso otimizado do Railway

Versão revisada do projeto original. **Mantém**: importação por link ou arquivo, detecção da troca dos dois takes, transcrição Whisper, OCR, texto manual, troca da primeira mídia, áudio original e renderização via FFmpeg.

## O que mudou

- **Novo layout** para desktop e celular, com personalização à esquerda e prévia/resultado à direita.
- **Três etapas:** original → personalizar → exportar.
- **Análise automática após importar um link ou enviar um vídeo.** A edição de texto não executa OCR/Whisper novamente.
- **Prévia leve** atualizada quando você troca foto/vídeo, texto, posição, tamanho ou perfil de qualidade. Usa uma imagem JPEG pequena, não FFmpeg completo.
- **A proporção de exportação segue o vídeo ORIGINAL.** Fotos e vídeos de outros formatos são encaixados com margem preta quando necessário.
- **Corte do vídeo e áudio sincronizados:** eliminado o salto extra de 0,07 s usado anteriormente apenas na imagem da continuação.
- **Exportação Padrão (até 1080 × 1920) ou Econômica (até 720 × 1280)**, até 30 FPS. Se o original tem resolução menor, não é ampliado.
- **Um player para o resultado e link para baixar MP4**, evitando carregar um player duplicado.
- **Limpeza de arquivos temporários** de resultados, imports, previews e cache Gradio somente quando há atividade; sem um processo permanente.
- **Renderização medida em logs** (tempo e MB de saída). Intermediários são removidos depois de exportação bem-sucedida.
- **API `/api/import-video` desativada por padrão**: pode ser habilitada com `IMPORT_API_TOKEN`, exigido como Bearer token.
- **Importador Instagram centralizado** em `core/instagram_import.py`; `url_import.py` mantém os nomes antigos como fachada de compatibilidade.

## Publicar no Railway

1. Envie os arquivos deste diretório para a **raiz do seu repositório GitHub**. A pasta `core/` deve permanecer na raiz, ao lado do `app.py` e do `Dockerfile`.
2. No Railway, conecte o repositório ou faça um novo deploy do serviço já existente.
3. Confira se o Railway está utilizando o `Dockerfile` e se a URL pública aponta para a porta definida automaticamente em `PORT`.
4. No serviço, em **Variables**, configure no mínimo `APP_USERNAME` e `APP_PASSWORD` com valores fortes e diferentes. Não coloque senhas no GitHub.
5. Em **Settings → Deploy → Serverless**, ative `Enable Serverless` e **faça um novo deploy**. O toggle aplica-se aos novos containers, não ao que já está em execução.
6. Vá até **Metrics** e **Workspace → Usage** para observar RAM, CPU e custos antes/depois. Configure alertas/limites de gasto no Railway.

> **IMPORTANTE — o Railway continuará cobrando a assinatura do plano, mesmo com o serviço dormindo.** O modo Serverless reduz o tempo de execução cobrado; não elimina a taxa mensal do plano. Uma primeira abertura após suspensão pode demorar ou retornar 502 temporariamente, tente recarregar. Consultar `/health`, sites externos de uptime e abas do navegador abertas também podem impedir a suspensão.

### Por que o Railway gastava sem visitas?

Sem Serverless, o processo Python/Gradio continua em execução e mantém RAM alocada mesmo quando você não está editando um vídeo. O Railway cobra recursos do serviço em execução, não apenas a quantidade de vídeos gerados.

O aplicativo **não inicia cron, timers de limpeza, verificação repetida de status ou telemetria voluntária**. Contudo, outras conexões, probes externos e o próprio uso do browser também podem mantê-lo acordado. A opção Serverless tem que ser ligada **manualmente** no painel Railway.

### Variáveis sugeridas

Há um arquivo [`env.example`](env.example) com todas as opções. Valores iniciais relevantes:

```env
APP_USERNAME=USUARIO_FORTE
APP_PASSWORD=SENHA_FORTE
MAX_UPLOAD_SIZE=200mb
MAX_VIDEO_MINUTES=10
MAX_INSTAGRAM_DOWNLOAD_MB=200
WHISPER_MODEL=tiny
WHISPER_KEEP_MODEL_LOADED=0
OUTPUT_PRESET=veryfast
OUTPUT_MAX_LONG_EDGE=1920
FFMPEG_THREADS=1
TEMP_MAX_AGE_HOURS=3
GRADIO_CACHE_MAX_AGE_HOURS=6
```

Se todos os vídeos forem curtos, `MAX_VIDEO_MINUTES=3` pode evitar processamento longo e acidental. `MAX_UPLOAD_SIZE` e `MAX_INSTAGRAM_DOWNLOAD_MB` limitam downloads/uploads excessivos. Para uso privado, **não deixe a interface sem login**: alguém pode consumir seus recursos usando a URL pública.

*Se estiver usando a importação pelo Instagram*, as variáveis opcionais `INSTAGRAM_COOKIES_B64` / `INSTAGRAM_COOKIES` continuam disponíveis. O Instagram pode bloquear downloads em IPs de data center, independentemente da versão do projeto.

### API opcional

A importação pela **interface web funciona sem configurar API externa**. A rota `POST /api/import-video` e a rota antiga `/midia/{job_id}.mp4` só podem ser usadas com `IMPORT_API_TOKEN`:

```http
Authorization: Bearer <SEU_TOKEN>
```

A rota `/health` continua pública para a verificação de deploy. As ações internas do Gradio não são publicadas como API pública na documentação.

## Usar a ferramenta

1. **Vídeo original:** cole o link do Reel e clique em **Importar e analisar**, ou use a aba **Enviar arquivo**. A análise inicia automaticamente.
2. **Personalizar:** envie a nova foto ou vídeo, revise o texto detectado e escolha posição/tamanho. A prévia é atualizada sem gerar o MP4 completo.
3. **Exportar:** selecione o perfil Padrão ou Econômico nas configurações avançadas, clique em **Gerar vídeo final** e depois **Baixar MP4**.

Se a análise não conseguir detectar com confiança onde começa o segundo take, abra **Opções avançadas**, escolha **Informar o segundo manualmente** e clique em **Reanalisar original**.

Quando o método de legenda é alterado, reanalise o original. Isso evita que a transcrição antiga seja reutilizada incorretamente.

## Rodar localmente

Requer Python 3.11, FFmpeg, FFprobe e Tesseract no PATH. Para habilitar OCR em português, instale também os idiomas do Tesseract (`por`, `eng`, `spa`). O Dockerfile já provisiona esses programas.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Acesse `http://localhost:7860`. No Windows, ative o ambiente virtual conforme sua instalação do Python. `run_local.bat` e `run_local.sh` do projeto original permanecem disponíveis.

## Testes

```bash
python -m pytest -q tests/test_studio_reels.py
```

Os testes incluem perfis de qualidade, precisão do corte de áudio/vídeo e uma renderização real de quatro segundos. Para isso, é necessário `pytest` no ambiente de desenvolvimento (não é instalado no Railway).

**Validação realizada nesta entrega:** testes de render local, inicialização da interface Gradio e rotas HTTP. A importação real de um link do Instagram e o deploy no seu Railway exigem rede/credenciais e precisam ser confirmados no seu ambiente. O arquivo ZIP entregue é o projeto completo, não uma atualização aplicada à sua conta Railway.

## Observação sobre compatibilidade

- O formato de saída agora acompanha o **original**, não a resolução da nova foto. Esse ajuste foi intencional.
- Os perfis podem reduzir vídeos de 60 para 30 FPS para reduzir custo. Para preservar 60 FPS seria necessário adaptar `export_settings` e aumentar a exigência computacional.
- Arquivos de resultados e prévias **são temporários**. Faça download antes de deixar o serviço inativo por muito tempo.
- Outros serviços, réplicas, volumes e a assinatura Railway podem gerar cobranças independentemente deste código.
