import json
import os
import re

# choose the jsonl file path
input_jsonl_file = ''
# chooce the output path
output_directory = '' 

output_text_file = os.path.join(output_directory, 'text')
output_wav_scp_file = os.path.join(output_directory, 'wav.scp')

def convert_jsonl():
    try:
        with open(output_text_file, 'w', encoding='utf-8') as text_file, \
             open(output_wav_scp_file, 'w', encoding='utf-8') as wav_scp_file:

            with open(input_jsonl_file, 'r', encoding='utf-8') as jsonl_file:
                for line in jsonl_file:
                    try:
                        data = json.loads(line.strip())

                        key = data.get('key')
                        source = data.get('source')
                        target = data.get('target')

                        if not all([key, source, target]):
                            print(f"error! {line.strip()}")
                            continue

                        match = re.search(r'common_voice_en_(\d+)\.mp3', key)
                        
                        file_id = match.group(1)

                        text_file.write(f"{file_id} {target}\n")

                        wav_scp_file.write(f"{file_id} {source}\n")

                    except json.JSONDecodeError:
                        print(f"warrning -> {line.strip()}")
                    except Exception as e:
                        print(f"warrning: {line.strip()} -> {e}")

        print("\ncomplete！")
        print(f"text file: {os.path.abspath(output_text_file)}")
        print(f"wav.scp file: {os.path.abspath(output_wav_scp_file)}")

    except FileNotFoundError:
        print(f"error> {input_jsonl_file}")
    except Exception as e:
        print(f"unknow error {e}")

if __name__ == '__main__':
    convert_jsonl()