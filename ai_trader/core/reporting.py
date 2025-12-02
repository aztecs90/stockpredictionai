from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import os
import io
from ai_trader.config.settings import REPORTS_DIR

class ReportGenerator:
    def __init__(self):
        pass

    def generate_pdf(self, ticker, stats, pf, filename="report.pdf"):
        filepath = os.path.join(REPORTS_DIR, filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph(f"AI Trading Report: {ticker}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Stats Table
        story.append(Paragraph("Performance Metrics", styles['Heading2']))

        # Convert Series to List of Lists for Table
        data = [['Metric', 'Value']]
        for idx, value in stats.items():
            # Format numbers
            if isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
            data.append([str(idx), val_str])

        # Only take top 20 metrics to fit page
        data = data[:25]

        table = Table(data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Plots
        story.append(Paragraph("Cumulative Returns", styles['Heading2']))

        # Generate Plot Image in Memory
        plt.figure(figsize=(8, 4))
        pf.plot().savefig("temp_plot.png")
        plt.close()

        img = Image("temp_plot.png", width=500, height=250)
        story.append(img)

        # Build PDF
        try:
            doc.build(story)
            if os.path.exists("temp_plot.png"):
                os.remove("temp_plot.png")
            return filepath
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
