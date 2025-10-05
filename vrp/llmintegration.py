import os
import google.generativeai as genai
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import logging
import json

# load .env file
load_dotenv()
# Configura a API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Inicializa o modelo
model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
def gerar_pdf_relatorio(dados_algoritmo, nome_arquivo="relatorio_genetico.pdf"):
    """
    Gera um PDF estilizado com base nos resultados do algoritmo genético.
    Inclui uma análise gerada pelo Gemini.
    """
    
    # Converte o dicionário de resultados do VRP em string JSON para análise pelo LLM
    dados_algoritmo_str = json.dumps(dados_algoritmo, indent=4, ensure_ascii=False)
    
    # 1. Configuração básica do PDF
    doc = SimpleDocTemplate(nome_arquivo, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Adiciona estilo para o JSON (para Preformatted)
    styles.add(ParagraphStyle(name='CodeStyle', fontName='Courier', fontSize=8, 
                              leading=10, leftIndent=10, rightIndent=10, textColor=colors.navy))
    
    # 2. Adiciona um título ao PDF
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, alignment=1, spaceAfter=20))
    title = Paragraph("Relatório de Otimização VRP (Algoritmo Genético)", styles['TitleStyle'])
    Story.append(title)

    # 3. Geração e Inclusão da Análise pelo Gemini
    try:
        # Prompt para o Gemini analisar o JSON dos resultados
        prompt = (
            "Você é um analista de otimização de rotas. Analise os seguintes resultados de um algoritmo genético VRP. "
            "Destaque o custo normalizado final (best_normalized_cost), o número de veículos usados, "
            "e avalie brevemente se os resultados são promissores, com base nas restrições fornecidas. "
            "Forneça sua análise como um parágrafo conciso, em português."
            f"\n\nResultados VRP (JSON): {dados_algoritmo_str}"
        )
        
        response = model.generate_content(prompt)
        analise_gemini = response.text
        
    except Exception as e:
        analise_gemini = f"Erro ao gerar análise do Gemini: {e}"
        logging.error(analise_gemini)

    # Adiciona a análise gerada
    Story.append(Paragraph("<b>Análise do Gemini:</b>", styles['Heading2']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(analise_gemini, styles['Normal'])) # Agora 'analise_gemini' é uma string
    Story.append(Spacer(1, 24))

    # 4. Adiciona os principais métricas em uma tabela
    Story.append(Paragraph("<b>Métricas Chave da Solução:</b>", styles['Heading2']))
    Story.append(Spacer(1, 12))

    # Prepara os dados para a tabela com base na estrutura real de final_output
    constraints = dados_algoritmo.get('base_constraints', {})
    
    table_data = [
        ['Métrica', 'Valor'],
        ['Custo Normalizado Final', f"{dados_algoritmo.get('best_normalized_cost', 'N/A'):.4f}"],
        ['Veículos Utilizados', dados_algoritmo.get('number_of_vehicles_used', 'N/A')],
        ['Frota Máxima Permitida', constraints.get('max_fleet_vehicles', 'N/A')],
        ['Paradas Máximas por Viagem', constraints.get('max_stops_per_trip', 'N/A')],
        ['Duração Máxima da Viagem (minutos)', constraints.get('max_duration_minutes', 'N/A')],
        ['URL OSRM Base', dados_algoritmo.get('cost_matrix_based_on', 'N/A')],
    ]
    
    t = Table(table_data, colWidths=[200, 300])
    
    # Estiliza a tabela
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDDDDD')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    
    Story.append(t)
    Story.append(Spacer(1, 24))

    # 5. Adiciona os dados JSON brutos (visualização do dado que gerou a análise)
    Story.append(Paragraph("<b>Dados Brutos da Solução (JSON):</b>", styles['Heading2']))
    Story.append(Spacer(1, 6))
    
    # Usa Preformatted para preservar a formatação JSON
    Story.append(Preformatted(dados_algoritmo_str, styles['CodeStyle']))
    Story.append(Spacer(1, 24))

    # 6. Constrói o PDF
    try:
        doc.build(Story)
        logging.info(f"PDF '{nome_arquivo}' gerado com sucesso!")
    except Exception as e:
        logging.error(f"Erro ao construir o PDF: {e}")

    # Retorna o JSON original, necessário no main.py
    return dados_algoritmo_str
