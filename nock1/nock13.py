pat = 'パトカー'
tax = 'タクシー'
ans = ''

for p, t in zip(pat, tax):
    ans += p+t

print(ans)