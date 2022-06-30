device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
X = X.to(device)
Y = Y.to(device)
X_v = X_v.to(device)
Y_v = Y_v.to(device)
X_t = X_t.to(device)
Y_t = Y_t.to(device)

model = network().to(device)
loss_fn = nn.CrossEntropyLoss()