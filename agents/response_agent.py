class ResponseAgent:
    def take_action(self, is_threat, data):
        src_ip = data.get('src_ip', 'Unknown')
        label = data.get('label', 'Unknown')

        if is_threat:
            print(f"[!!] ALERT: Malicious activity detected! Source IP: {src_ip}, Label: {label}")
            print("     → Simulating IP block.\n")
        else:
            print(f"[OK] Normal traffic from {src_ip}. No action taken.\n")
