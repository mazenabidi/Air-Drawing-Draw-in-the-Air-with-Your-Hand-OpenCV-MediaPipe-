import mediapipe as mp
print(dir(mp))
if hasattr(mp, 'tasks'):
    print("\nmp.tasks available!")
    print(dir(mp.tasks))
if hasattr(mp, 'solutions'):
    print("\nmp.solutions available!")
    print(dir(mp.solutions))
