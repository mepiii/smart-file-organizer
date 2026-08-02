import os, argparse
from src.trainer import train_model
from src.organizer import SmartFileOrganizer

def main():
    b = os.path.dirname(os.path.abspath(__file__))
    d, m, c = os.path.join(b, 'dataset', 'file_history.csv'), os.path.join(b, 'models', 'file_organizer_model.joblib'), os.path.join(b, 'config', 'config.json')
    if not os.path.exists(m): train_model(d, m)
    org = SmartFileOrganizer(m, c)
    print('Smart File Organizer Demo:')
    for fname in ['CS101_Exam.pdf', 'Tax_Invoice.pdf', 'Design_UI.png', 'app_server.py']:
        folder, conf = org.predict_folder(fname)
        print(f'  {fname:<20} -> {folder:<20} ({conf*100:.1f}%)')

if __name__ == '__main__': main()
