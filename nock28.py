f = open('col1.txt', 'r')

data = f.read()
data_s = data.split()
set = []

for i in data_s:
    if i not in set:
        set.append(i)

print(set)
f.close()


"""
実行結果:
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



"""