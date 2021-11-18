def Program(ast):
    return ast['body']


def ExpressionStatement(ast):
    return [ast['expression']]


def FunctionDeclaration(ast):
    return [ast['body']]


def AssignmentExpression(ast):
    return [ast['left'], ast['right']]


def BlockStatement(ast):
    return ast['body']


def CallExpression(ast):
    return ast['arguments']


def MemberExpression(ast):
    return [ast['object'], ast['property']]


def LogicalExpression(ast):
    return [ast['left'], ast['right']]


def Literal(ast):
    return []


def NewExpression(ast):
    return ast['arguments']


def Identifier(ast):
    return []


def ArrayExpression(ast):
    return []


def VariableDeclaration(ast):
    return ast['declarations']


def VariableDeclarator(ast):
    return [ast['id'], ast['init']]


def FunctionExpression(ast):
    return [ast['body']]


def BinaryExpression(ast):
    return [ast['left'], ast['right']]


def ObjectExpression(ast):
    return ast['properties']


def Property(ast):
    return [ast['key'], ast['value']]


def ForStatement(ast):
    return [ast['body'], ast['init'], ast['test'], ast['update']]


def UpdateExpression(ast):
    return [ast['argument']]


def ConditionalExpression(ast):
    return [ast['alternate'], ast['consequent'], ast['test']]


def TryStatement(ast):
    result = [ast['block']]
    result.extend(ast['handlers'])
    return result


def CatchClause(ast):
    return [ast['body'], ast['param']]


def IfStatement(ast):
    return [ast['alternate'], ast['consequent'], ast['test']]


def BreakStatement(ast):
    return []


def UnaryExpression(ast):
    return [ast['argument']]


def EmptyStatement(ast):
    return []


def ReturnStatement(ast):
    return [ast['argument']]


def DoWhileStatement(ast):
    return [ast['body'], ast['test']]


def WhileStatement(ast):
    return [ast['body'], ast['test']]


def WithStatement(ast):
    return [ast['body'], ast['object']]


def ThisExpression(ast):
    return []


def SequenceExpression(ast):
    return ast['expressions']


def ForInStatement(ast):
    return [ast['body'], ast['left'], ast['right']]


def SwitchStatement(ast):
    result = [ast['discriminant']]
    result.extend(ast['cases'])
    return result


def SwitchCase(ast):
    result = [ast['test']]
    result.extend(ast['consequent'])
    return result


def ThrowStatement(ast):
    return [ast['argument']]


def LabeledStatement(ast):
    return [ast['body']]


def ContinueStatement(ast):
    return []


def DebuggerStatement(ast):
    return []