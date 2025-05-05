n = int(input())
actions = [input().strip() for _ in range(n)]

apps = []
result = []

for action in actions:
    if action.startswith('Run'):
        if len(action) == 4:
            app_name = ' '
        else:
            app_name = action[4:]
        apps.insert(0, app_name)
        result.append(app_name)
    elif action.startswith('Alt'):
        if not apps:
            continue
        tab_count = action.count('+Tab')
        k = tab_count % len(apps)
        selected_app = apps[k]
        apps.pop(k)
        apps.insert(0, selected_app)
        result.append(selected_app)

print('\n'.join(result))