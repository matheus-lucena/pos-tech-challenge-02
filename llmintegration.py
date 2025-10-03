import os
import google.generativeai as genai
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# load .env file
load_dotenv()
# Configura a API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Inicializa o modelo
model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
def gerar_pdf_relatorio(dados_algoritmo, nome_arquivo="relatorio_genetico.pdf"):
    """
    Gera um PDF estilizado com base nos resultados do algoritmo genético.
    """
    
    # 1. Configuração básica do PDF
    doc = SimpleDocTemplate(nome_arquivo, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # 2. Adiciona um título ao PDF
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, alignment=1, spaceAfter=20))
    title = Paragraph("Relatório do Algoritmo Genético", styles['TitleStyle'])
    Story.append(title)

    # 3. Adiciona a análise gerada pelo Gemini
    Story.append(Paragraph("<b>Análise do Gemini:</b>", styles['Normal']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(dados_algoritmo, styles['Normal']))
    Story.append(Spacer(1, 24))

    # 4. Adiciona os detalhes dos resultados em uma tabela
    Story.append(Paragraph("<b>Detalhes dos Resultados:</b>", styles['Normal']))
    Story.append(Spacer(1, 12))

    # Prepara os dados para a tabela
    table_data = [['Parâmetro', 'Melhor Valor', 'Pontuação de Aptidão']]
    for key, value in dados_algoritmo['melhor_individuo'].items():
        table_data.append([key, str(value), str(dados_algoritmo['melhor_fitness'])])

    t = Table(table_data)
    
    # Estiliza a tabela
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    
    Story.append(t)

    # 5. Constrói o PDF
    doc.build(Story)
    print(f"PDF '{nome_arquivo}' gerado com sucesso!")
