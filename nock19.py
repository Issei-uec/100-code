def cipher(str):
    code = ''
    for i in str:
        if i.islower():
            code += chr(219-ord(i))
        else:
            code += i   
    return code

original_sentense = input('原文：')
print('暗号化：', cipher(original_sentense))
code_sentense = cipher(original_sentense)
print('複合化：', cipher(code_sentense))