# Registro técnico — Studio Reels 2.0

Base: `GERADOR-DE-POST-MODELO-IA-main (7).zip` (arquivos fornecidos pelo usuário).

## Interface
- Novo painel em duas colunas, cartões mais compactos, visual escuro no cabeçalho e texto claro.
- Upload e importação por URL em abas; importação inicia análise automática.
- Prévia adaptada às dimensões de exportação e atualizada nas alterações relevantes.
- Área de configurações avançadas recolhível, download direto e botão limpar.

## Processamento
- Perfil econômico em 720p, CRF 23 e áudio 128 kb/s; perfil padrão em 1080p, CRF ambiental e áudio ambiental.
- Resolução de saída derivada do vídeo original; nova mídia centralizada com barras pretas se proporções diferentes.
- Eliminação do salto extra de 70ms no vídeo, alinhando a continuação com o áudio que já era preservado integralmente.
- 30 FPS máximos nos perfis atuais.
- Remoção de arquivos temporários intermediários em caso de sucesso e registro do tempo de processamento.

## Railway e segurança
- Sem loops de cleanup ou heartbeat em segundo plano; coleta oportunista de temporários.
- `GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_DISABLE_TELEMETRY=1`, API fechada por padrão com `IMPORT_API_TOKEN` opcional.
- API Gradio interna marcada como `private`; autenticação da interface via `APP_USERNAME` e `APP_PASSWORD`.
- `.dockerignore` e `url_import.py` sem lógica duplicada.
- **O toggle Serverless é manual no Railway.** Não é possível prová-lo ligado a partir deste ZIP.

## Verificações executadas
- `python -m compileall` para sintaxe.
- `pytest` com teste de render completo e validação de saída MP4 com áudio.
- Gradio 6.5.1 importada e montada com sucesso, `GET /` e `GET /health` responderam 200; `POST /api/import-video` sem token retornou 401.
- Prévia JPEG gerada e exibível.
- Sem teste de conexão real com Instagram, deploy Railway ou teste visual no navegador do usuário.
