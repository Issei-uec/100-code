m_file = open('col1.txt', 'r')

data = m_file.read()
data_s = data.split()

namelist = []
for name in data_s:
    if name not in namelist:
        namelist.append(name)

count = 0
for name in namelist:
    count += 1

print(namelist)
print(count)

m_file.close()


"""
実行結果:
136
['Mary', 'Anna', 'Emma', 'Elizabeth', 'Minnie', 'Margaret', 'Ida', 'Alice', 'Bertha', 'Sarah', 'John', 'William', 'James', 'Charles', 'George', 'Frank', 'Joseph', 'Thomas', 'Henry', 'Robert', 'Annie', 'Edward', 'Clara', 'Florence', 'Ethel', 'Bessie', 'Harry', 'Helen', 'Ruth', 'Marie', 'Lillian', 'Mildred', 'Dorothy', 'Frances', 'Walter', 'Evelyn', 'Virginia', 'Richard', 'Betty', 'Donald', 'Doris', 'Shirley', 'Barbara', 'Patricia', 'Joan', 'Nancy', 'Carol', 'David', 'Ronald', 'Judith', 'Linda', 'Sandra', 'Carolyn', 'Sharon', 'Michael', 'Susan', 'Donna', 'Larry', 'Kathleen', 'Deborah', 'Gary', 'Karen', 'Debra', 'Pamela', 'Cynthia', 'Mark', 'Steven', 'Lisa', 'Jeffrey', 'Lori', 'Kimberly', 'Tammy', 'Angela', 'Michelle', 'Jennifer', 'Melissa', 'Christopher', 'Brian', 'Amy', 'Laura', 'Tracy', 'Julie', 'Jason', 'Scott', 'Stephanie', 'Heather', 'Nicole', 'Matthew', 'Rebecca', 'Jessica', 'Amanda', 'Daniel', 'Kelly', 'Joshua', 'Crystal', 'Ashley', 'Megan', 'Brittany', 'Andrew', 'Justin', 'Samantha', 'Lauren', 'Emily', 'Brandon', 'Tyler', 'Taylor', 'Nicholas', 'Jacob', 'Hannah', 'Austin', 'Alexis', 'Rachel', 'Madison', 'Abigail', 'Olivia', 'Ethan', 'Anthony', 'Isabella', 'Ava', 'Sophia', 
'Chloe', 'Alexander', 'Mia', 'Jayden', 'Noah', 'Aiden', 'Mason', 'Liam', 'Charlotte', 'Harper', 'Benjamin', 'Elijah', 'Amelia', 'Logan', 'Oliver', 'Lucas']

UNIX:
cut -f 1 -d " " test.txt > 17.txt
sort 17.txt > 17_sort.txt
uniq 17_sort.txt

実行結果:
Abigail
Aiden
Alexander
Alexis
Alice

リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""