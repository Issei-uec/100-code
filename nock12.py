str = 'パタトクカシーー'
ans = ''
for i in range(len(str)):
    if i % 2 == 1:
        ans += str[i]

print(ans)

ans1 = str[0:6:2]