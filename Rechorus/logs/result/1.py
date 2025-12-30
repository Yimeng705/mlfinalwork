import re
# 匹配独占一行的空白行（包括只有空格的行）
pattern = r'^\s*$'
# 使用 MULTILINE 标志使 ^ 和 $ 匹配每行的开始和结束
result = re.sub(pattern, '', text, flags=re.MULTILINE)