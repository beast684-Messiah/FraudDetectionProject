import json
import os
from datetime import datetime

class HistoryDatabase:
    def __init__(self):
        self.file_path = 'model/history.json'
        self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_history(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def add_record(self, form_data, prediction, risk_level):
        record = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'form_data': form_data,
            'prediction': prediction,
            'risk_level': risk_level
        }
        self.history.append(record)
        self.save_history()
        return record['id']

    def get_all_records(self):
        return sorted(self.history, key=lambda x: x['timestamp'], reverse=True)

    def get_record_by_id(self, record_id):
        for record in self.history:
            if record['id'] == record_id:
                return record
        return None
        
    def delete_record(self, record_id):
        """Delete a record by its ID"""
        for i, record in enumerate(self.history):
            if record['id'] == record_id:
                self.history.pop(i)
                self.save_history()
                return True
        return False 