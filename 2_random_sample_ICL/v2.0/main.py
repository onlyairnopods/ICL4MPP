import subprocess
import time

def run_script(script_name, idx):
    try:
        # 捕获标准输出和错误输出
        result = subprocess.run(['python', script_name, '--params_idx', str(idx)], check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        return False
    return False

params_idx_file = './log/params_idx.txt'
fail_params_idx_file = './log/fail_params_idx.txt'

while True:
    idx = int(open(params_idx_file, 'r').read().strip())  # 使用strip()来去除可能的空白字符
    print(idx)
    if idx > 54:
        break

    if not run_script('1_random_ICL.py', idx):
        with open(fail_params_idx_file, 'a') as f:
            f.write(str(idx) + '\n')

        # 如果运行失败，增加params_idx的值
        with open(params_idx_file, 'w') as f:
            f.write(str(idx + 1))

    time.sleep(5)