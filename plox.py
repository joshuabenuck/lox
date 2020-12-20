import sys, time
from enum import Enum

TokenType = Enum('TokenType', 
    # Single character tokens
    'LEFT_PAREN,RIGHT_PAREN,LEFT_BRACE,RIGHT_BRACE,'
    'COMMA,DOT,MINUS,PLUS,SEMICOLON,SLASH,STAR,'

    # One or two character tokens
    'BANG,BANG_EQUAL,EQUAL,EQUAL_EQUAL,'
    'GREATER,GREATER_EQUAL,LESS,LESS_EQUAL,'

    # Literals
    'IDENTIFIER,STRING,NUMBER,'

    # Keywords
    'AND,CLASS,ELSE,FALSE,FUN,FOR,IF,NIL,OR,'
    'PRINT,RETURN,SUPER,THIS,TRUE,VAR,WHILE,'

    'EOF'
)

class Token(object):
    def __init__(self, type: TokenType, lexeme: str, literal, line):
        self.type = type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __repr__(self):
        return "{} {} {}".format(self.type, self.lexeme, self.literal)

class Scanner(object):
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.keywords = {
            'and':    TokenType.AND,
            'class':  TokenType.CLASS,
            'else':   TokenType.ELSE,
            'false':  TokenType.FALSE,
            'for':    TokenType.FOR,
            'fun':    TokenType.FUN,
            'if':     TokenType.IF,
            'nil':    TokenType.NIL,
            'or':     TokenType.OR,
            'print':  TokenType.PRINT,
            'return': TokenType.RETURN,
            'super':  TokenType.SUPER,
            'this':   TokenType.THIS,
            'true':   TokenType.TRUE,
            'var':    TokenType.VAR,
            'while':  TokenType.WHILE
        }

    def scanTokens(self):
        while not self.isAtEnd():
            # We are at the beginning of the next lexeme
            self.start = self.current
            self.scanToken()
        
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    def isAtEnd(self):
        return self.current >= len(self.source)

    def scanToken(self):
        c = self.advance()
        if c == '(':
            self.addToken(TokenType.LEFT_PAREN)
        elif c == ')':
            self.addToken(TokenType.RIGHT_PAREN)
        elif c == '{':
            self.addToken(TokenType.LEFT_BRACE)
        elif c == '}':
            self.addToken(TokenType.RIGHT_BRACE)
        elif c == ',':
            self.addToken(TokenType.COMMA)
        elif c == '.':
            self.addToken(TokenType.DOT)
        elif c == '-':
            self.addToken(TokenType.MINUS)
        elif c == '+':
            self.addToken(TokenType.PLUS)
        elif c == ';':
            self.addToken(TokenType.SEMICOLON)
        elif c == '*':
            self.addToken(TokenType.STAR)
        elif c == '!':
            if self.match('='):
                self.addToken(TokenType.BANG_EQUAL)
            else:
                self.addToken(TokenType.BANG)
        elif c == '=':
            if self.match('='):
                self.addToken(TokenType.EQUAL_EQUAL)
            else:
                self.addToken(TokenType.EQUAL)
        elif c == '<':
            # TODO: Change to LESSER?
            if self.match('='):
                self.addToken(TokenType.LESS_EQUAL)
            else:
                self.addToken(TokenType.LESS)
        elif c == '>':
            if self.match('='):
                self.addToken(TokenType.GREATER_EQUAL)
            else:
                self.addToken(TokenType.GREATER)
        elif c == '/':
            if self.match('/'):
                # A comment goes until the end of the line.
                while self.peek() != '\n' and not self.isAtEnd(): self.advance()
            elif self.match('*'):
                while self.peek() != '*' and self.peekNext() != '/' and not self.isAtEnd(): self.advance()
                # Consume the */ at the end
                self.advance()
                self.advance()
            else:
                self.addToken(TokenType.SLASH)
        elif c == ' ' or c == '\r' or c == '\t':
            pass
        elif c == '\n':
            self.line += 1
        elif c == '"':
            self.string()
        else:
            if self.isDigit(c):
                self.number()
            elif self.isAlpha(c):
                self.identifier()
            else:
                error(self.line, "Unexpected character.")

    def string(self):
        while self.peek() != '"' and not self.isAtEnd():
            if self.peek() == '\n': self.line+=1
            self.advance()

        if self.isAtEnd():
            error(self.line, "Unterminated string.")
            return

        self.advance()

        value = self.source[self.start+1:self.current-1]
        self.addToken(TokenType.STRING, value)

    def isDigit(self, c):
        return c >= '0' and c <= '9'

    def number(self):
        while self.isDigit(self.peek()): self.advance()

        if self.peek() == '.' and self.isDigit(self.peekNext()):
            self.advance()
            while self.isDigit(self.peek()): self.advance()

        self.addToken(TokenType.NUMBER,
            float(self.source[self.start:self.current]))

    def identifier(self):
        while self.isAlphaNumeric(self.peek()):
            self.advance()

        text = self.source[self.start:self.current]
        type = None
        if text in self.keywords:
            type = self.keywords[text]
        if type == None:
            type = TokenType.IDENTIFIER
        self.addToken(type)

    def isAlpha(self, c):
        return c >= 'a' and c <= 'z' or \
            c >= 'A' and c <= 'Z' or \
            c == '_'

    def isAlphaNumeric(self, c):
        return self.isAlpha(c) or self.isDigit(c)

    def advance(self):
        self.current+=1
        return self.source[self.current-1]

    def match(self, expected):
        if self.isAtEnd(): return False
        if self.source[self.current] != expected: return False
        self.current+=1
        return True

    def peek(self):
        if self.isAtEnd(): return '\0'
        return self.source[self.current]

    def peekNext(self):
        if self.current + 1 > len(self.source):
            return '\0'
        return self.source[self.current + 1]

    def addToken(self, type, literal=None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line))

class Expr(object):
    pass

class Assign(Expr):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def accept(self, visitor):
        return visitor.visitAssignExpr(self)

class Binary(Expr):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visitBinaryExpr(self)

class Call(Expr):
    def __init__(self, callee, paren, arguments):
        self.callee = callee
        self.paren = paren
        self.arguments = arguments

    def accept(self, visitor):
        return visitor.visitCallExpr(self)

class Unary(Expr):
    def __init__(self, operator, right):
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visitUnaryExpr(self)

class Variable(Expr):
    def __init__(self, name):
        self.name = name

    def accept(self, visitor):
        return visitor.visitVariableExpr(self)

class Literal(Expr):
    def __init__(self, value):
        self.value = value

    def accept(self, visitor):
        return visitor.visitLiteralExpr(self)

class Logical(Expr):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visitLogicalExpr(self)

class Grouping(Expr):
    def __init__(self, expr):
        self.expr = expr

    def accept(self, visitor):
        return visitor.visitGroupingExpr(self)

class Stmt(object):
    pass

class Block(Stmt):
    def __init__(self, stmts):
        self.stmts = stmts

    def accept(self, visitor):
        return visitor.visitBlockStmt(self)

class Expression(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def accept(self, visitor):
        return visitor.visitExpressionStmt(self)

class Function(Stmt):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

    def accept(self, visitor):
        return visitor.visitFunctionStmt(self)

class If(Stmt):
    def __init__(self, condition, thenBranch, elseBranch):
        self.condition = condition
        self.thenBranch = thenBranch
        self.elseBranch = elseBranch

    def accept(self, visitor):
        return visitor.visitIfStmt(self)

class Print(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def accept(self, visitor):
        return visitor.visitPrintStmt(self)

class Return(Stmt):
    def __init__(self, keyword, value):
        self.keyword = keyword
        self.value = value

    def accept(self, visitor):
        return visitor.visitReturnStmt(self)

class Var(Stmt):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer

    def accept(self, visitor):
        return visitor.visitVarStmt(self)

class While(Stmt):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def accept(self, visitor):
        return visitor.visitWhileStmt(self)

class Parser(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        statements = []
        while not self.isAtEnd():
            statements.append(self.declaration())
        return statements

    def declaration(self):
        try:
            if self.match(TokenType.FUN):
                return self.function("function")
            if self.match(TokenType.VAR):
                return self.varDeclaration()

            return self.statement()
        except ParseError as error:
            self.synchronize()
            return None

    def statement(self):
        if self.match(TokenType.FOR):
            return self.forStatement()
        if self.match(TokenType.IF):
            return self.ifStatement()
        if self.match(TokenType.PRINT):
            return self.printStatement()
        if self.match(TokenType.RETURN):
            return self.returnStatement()
        if self.match(TokenType.WHILE):
            return self.whileStatement()
        if self.match(TokenType.LEFT_BRACE):
            # wrapped here so block can be reused to parse function bodies
            return Block(self.block())

        return self.expressionStatement()

    def forStatement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'for'.")

        initializer = None
        if self.match(TokenType.SEMICOLON):
            initializer = None
        elif self.match(TokenType.VAR):
            initializer = self.varDeclaration()
        else:
            initializer = self.expressionStatement()

        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after loop condition.")

        increment = None
        if not self.check(TokenType.RIGHT_PAREN):
            increment = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after for clauses.")

        body = self.statement()

        if increment != None:
            body = Block([body, Expression(increment)])

        if condition == None:
            condition = Literal(True)

        body = While(condition, body)

        if initializer != None:
            body = Block([initializer, body])

        return body

    def ifStatement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'if'.")
        condition = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after if condition.")

        thenBranch = self.statement()
        elseBranch = None
        if self.match(TokenType.ELSE):
            elseBranch = self.statement()

        return If(condition, thenBranch, elseBranch)

    def printStatement(self):
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after value.")
        return Print(value)

    def returnStatement(self):
        keyword = self.previous()
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()

        self.consume(TokenType.SEMICOLON, "Expect ';' after return value.")
        return Return(keyword, value)

    def varDeclaration(self):
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name.");

        initializer = None
        if self.match(TokenType.EQUAL):
            initializer = self.expression()

        self.consume(TokenType.SEMICOLON, "Expect ';' after variable declaration.")
        return Var(name, initializer)

    def whileStatement(self):
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'while'.")
        condition = self.expression()
        self.consume(TokenType.LEFT_PAREN, "Expect ')' after condition.")

        body = self.statement()

        return While(condition, body)

    def expressionStatement(self):
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after expression.")
        return Expression(expr)

    def function(self, kind):
        name = self.consume(TokenType.IDENTIFIER, "Expect {} name.".format(kind))
        self.consume(TokenType.LEFT_PAREN, "Expect '(' after {} name.".format(kind))
        parameters = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(parameters) >= 255:
                    self.error(self.peek(), "Can't have more than 255 parameters.")

                parameters.append(self.consume(TokenType.IDENTIFIER, "Expect parameter name."))
                if not self.match(TokenType.COMMA):
                    break
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after parameters.")

        self.consume(TokenType.LEFT_BRACE, "Expect '{{' before {} body.".format(kind))
        body = self.block()
        return Function(name, parameters, body)

    def block(self):
        statements = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.isAtEnd():
            statements.append(self.declaration())
        self.consume(TokenType.RIGHT_BRACE, "Expect '}' after block.")

        return statements

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.orExpression()

        if self.match(TokenType.EQUAL):
            equals = self.previous()
            value = self.assignment()

            if type(expr) == Variable:
                name = expr.name
                return Assign(name, value)

            error(equals, "Invalid assignment target.")

        return expr

    def orExpression(self):
        expr = self.andExpression()

        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.andExpression()
            expr = Logical(expr, operator, right)

        return expr

    def andExpression(self):
        expr = self.equality()

        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.equality()
            expr = Logical(expr, operator, right)

        return expr

    def equality(self):
        expr = self.comparison()

        while self.match(TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = Binary(expr, operator, right)

        return expr

    def comparison(self):
        expr = self.term()

        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL,
            TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous()
            right = self.term()
            expr = Binary(expr, operator, right)

        return expr

    def term(self):
        expr = self.factor()

        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)

        return expr

    def factor(self):
        expr = self.unary()

        while self.match(TokenType.SLASH, TokenType.STAR):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)

        return expr

    def unary(self):
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)

        return self.call()

    def finishCall(self, callee):
        arguments = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(arguments) >= 255:
                    self.error(self.peek(), "Can't have more than 255 arguments.")
                arguments.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break

        paren = self.consume(TokenType.RIGHT_PAREN, "Expect ')' after arguments.")
        return Call(callee, paren, arguments)

    def call(self):
        expr = self.primary()

        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finishCall(expr)
            else:
                break

        return expr

    def primary(self):
        if self.match(TokenType.FALSE):
            return Literal(False)
        if self.match(TokenType.TRUE):
            return Literal(True)
        if self.match(TokenType.NIL):
            return Literal(None)

        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)

        if self.match(TokenType.IDENTIFIER):
            return Variable(self.previous())

        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return Grouping(expr)

        raise self.error(self.peek(), "Expect expression.")

    def match(self, *types):
        for type in types:
            if self.check(type):
                self.advance()
                return True

        return False

    def check(self, type):
        if self.isAtEnd():
            return False

        return self.peek().type == type

    def advance(self):
        if not self.isAtEnd():
            self.current += 1

        return self.previous()

    def isAtEnd(self):
        return self.peek().type == TokenType.EOF

    def peek(self):
        return self.tokens[self.current]

    def previous(self):
        return self.tokens[self.current - 1]

    def consume(self, type, message):
        if self.check(type):
            return self.advance()

        raise self.error(self.peek(), message)

    def error(self, token, message):
        error(token, message)
        return ParseError()

    def synchronize(self):
        self.advance()

        while not self.isAtEnd():
            if self.previous().type == TokenType.SEMICOLON:
                return
            if self.peek().type in [
                TokenType.CLASS,
                TokenType.FUN,
                TokenType.VAR,
                TokenType.FOR,
                TokenType.IF,
                TokenType.WHILE,
                TokenType.PRINT,
                TokenType.RETURN,
            ]:
                return
            self.advance()

class ParseError(Exception):
    pass

class Visitor(object):
    pass

class AstPrinter(Visitor):
    def print(self, expr):
        return expr.accept(self)

    def visitBinaryExpr(self, expr):
        return self.parenthesize(expr.operator.lexeme, expr.left, expr.right)

    def visitGroupingExpr(self, expr):
        return self.parenthesize("group", expr.expr)

    def visitLiteralExpr(self, expr):
        if expr.value == None:
            return "nil"
        return str(expr.value)

    def visitUnaryExpr(self, expr):
        return self.parenthesize(expr.operator.lexeme, expr.right)

    def parenthesize(self, name, *exprs):
        builder = ""
        builder += "(" + name
        for expr in exprs:
            builder += " "
            builder += expr.accept(self)
        builder += ")"
        return builder
    
class Interpreter(Visitor):
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        class ClockCallable(Callable):
            def arity(self): return 0
            def call(self, interpreter, arguments):
                return time.time()
            def __repr__(self):
                return "<native clock fn>"

        self.globals.define("clock", ClockCallable())

    def interpret(self, stmts):
        try:
            for stmt in stmts:
                self.execute(stmt)
        except RuntimeException as error:
            runtimeError(error)

    def execute(self, stmt):
        stmt.accept(self)

    def executeBlock(self, stmts, environment):
        previous = self.environment
        try:
            self.environment = environment

            for stmt in stmts:
                self.execute(stmt)
        finally:
            self.environment = previous

    def visitBlockStmt(self, stmt):
        self.executeBlock(stmt.stmts, Environment(self.environment))
        return None

    def evaluate(self, expr):
        return expr.accept(self)

    def visitExpressionStmt(self, stmt):
        self.evaluate(stmt.expr)
        return None

    def visitFunctionStmt(self, stmt):
        function = LoxFunction(stmt, self.environment)
        self.environment.define(stmt.name.lexeme, function)
        return None

    def visitIfStmt(self, stmt):
        if self.isTruthy(self.evaluate(stmt.condition)):
            self.execute(stmt.thenBranch)
        elif stmt.elseBranch != None:
            self.execute(stmt.elseBranch)
        return None

    def visitPrintStmt(self, stmt):
        value = self.evaluate(stmt.expr)
        print(self.stringify(value))
        return None

    def visitReturnStmt(self, stmt):
        value = None
        if stmt.value != None:
            value = self.evaluate(stmt.value)

        raise ReturnValueException(value)

    def visitVarStmt(self, stmt):
        value = None
        if stmt.initializer != None:
            value = self.evaluate(stmt.initializer)

        self.environment.define(stmt.name.lexeme, value)
        return None

    def visitWhileStmt(self, stmt):
        while self.isTruthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

        return None

    def visitAssignExpr(self, expr):
        value = self.evaluate(expr.value)
        self.environment.assign(expr.name, value)
        return value

    def visitBinaryExpr(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)

        if expr.operator.type == TokenType.GREATER:
            self.checkNumberOperands(expr.operator, left, right)
            return left > right
        if expr.operator.type == TokenType.GREATER_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return left >= right
        if expr.operator.type == TokenType.LESS:
            self.checkNumberOperands(expr.operator, left, right)
            return left < right
        if expr.operator.type == TokenType.LESS_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return left <= right
        if expr.operator.type == TokenType.BANG_EQUAL:
            return not self.isEqual(left, right)
        if expr.operator.type == TokenType.EQUAL_EQUAL:
            return self.isEqual(left, right)
        if expr.operator.type == TokenType.MINUS:
            self.checkNumberOperands(expr.operator, left, right)
            return left - right
        if expr.operator.type == TokenType.PLUS:
            # Not necessary, but leaving in anyway.
            if type(left) == type(right) == type(0.0):
                return left + right
            if type(left) == type(right) == type(""):
                return left + right
            raise RuntimeException(expr.operator, "Operands must be two numbers or two strings.")
        if expr.operator.type == TokenType.SLASH:
            self.checkNumberOperands(expr.operator, left, right)
            return left / right
        if expr.operator.type == TokenType.STAR:
            self.checkNumberOperands(expr.operator, left, right)
            return left * right

        # Unreachable
        return None

    def visitCallExpr(self, expr):
        callee = self.evaluate(expr.callee)

        arguments = []
        for argument in expr.arguments:
            arguments.append(self.evaluate(argument))

        # if not callee instanceof Callable:
        #   raise RuntimeException(expr.paren, "Can only call functions and classes."
        function = callee
        if len(arguments) != function.arity():
            raise RuntimeException(expr.paren, "Expected {} arguments, but got {}.".format(function.arity(), arguments.size()))
        return function.call(self, arguments)

    def visitGroupingExpr(self, expr):
        return self.evaluate(expr.expr)

    def visitLiteralExpr(self, expr):
        return expr.value

    def visitLogicalExpr(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)

        if expr.operator.type == TokenType.OR:
            if self.isTruthy(left):
                return left
        else:
            if not self.Truthy(left):
                return left

        self.evaluate(expr.right)

    def visitUnaryExpr(self, expr):
        right = self.evaluate(expr.right)

        if expr.operator.type == TokenType.BANG:
            return not self.isTruthy(right)
        if expr.operator.type == TokenType.MINUS:
            self.checkNumberOperand(expr.operator, right)
            return -right

        # Unreachable
        return None

    def visitVariableExpr(self, expr):
        return self.environment.get(expr.name)

    def checkNumberOperand(self, operator, operand):
        if type(operand) == type(0.0):
            return
        raise RuntimeException(operator, "Operand must be a number.")

    def checkNumberOperands(self, operator, left, right):
        if type(left) == type(right) == type(0.0):
            return
        raise RuntimeException(operator, "Operands must be numbers.")

    def isTruthy(self, obj):
        if obj == None:
            return False

        if type(obj) == type(False):
            return obj

        return True
   
    def isEqual(self, a, b):
        # Python semantics are different
        # This is another case where this could be simplified
        if a == None and b == None:
            return True
        if a == None:
            return False

        return a == b

    def stringify(self, obj):
        if obj == None:
            return "nil"

        if type(obj) == type(0.0):
            text = str(obj)
            if text[-2:] == ".0":
                text = text[0:len(text) - 2]
            return text

        return str(obj)

class RuntimeException(Exception):
    def __init__(self, token, message):
        super().__init__(message)
        self.token = token

class ReturnValueException(Exception):
    def __init__(self, value):
        super().__init__(self)
        self.value = value

class Environment(object):
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def get(self, name):
        if name.lexeme in self.values:
            return self.values[name.lexeme]

        if self.enclosing:
            return self.enclosing.get(name)

        raise RuntimeException(name, "Undefined variable '{}'".format(name.lexeme))

    def assign(self, name, value):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return

        if self.enclosing:
            self.enclosing.assign(name, value)
            return

        raise RuntimeException(name, "Undefined variable '{}'".format(name.lexeme))

    def define(self, name, value):
        self.values[name] = value

class Callable(object):
    # def call(self, interpreter, arguments): pass
    # def arity(self): pass
    pass

class LoxFunction(Callable):
    def __init__(self, declaration, closure):
        self.closure = closure
        self.declaration = declaration

    def arity(self):
        return len(self.declaration.params)

    def call(self, interpreter, arguments):
        environment = Environment(self.closure)
        for (i, param) in enumerate(self.declaration.params):
            environment.define(param.lexeme, arguments[i])

        try:
            interpreter.executeBlock(self.declaration.body, environment)
        except ReturnValueException as returnValue:
            return returnValue.value
        return None

    def __repr__(self):
        return "<fn {}>".format(self.declaration.name.lexeme)

def runFile(path: str):
    global hadError, hadRuntimeError
    with open(path) as file:
        run(file.read())
        if hadError: sys.exit(65)
        if hadRuntimeError: sys.exit(70)

def runPrompt():
    global hadError
    while True:
        line = input("> ")
        if len(line) == 0:
            break
        run(line)
        hadError = False

def run(source: str):
    global hadError, interpreter
    scanner = Scanner(source)
    tokens = scanner.scanTokens()

    parser = Parser(tokens)
    stmts = parser.parse()

    if hadError:
        return

    interpreter.interpret(stmts)

def error(token, message):
    if token.type == TokenType.EOF:
        report(token.line, " at end", message)
    else:
        report(token.line, " at '{}'".format(token.lexeme), message)

def error(line, message: str):
    report(line, "", message)

def runtimeError(error):
    global hadRuntimeError
    print("{}\n[line {}]".format(error, error.token.line))
    hadRuntimeError = True

def report(line, where: str, message: str):
    sys.stderr.write("[line {}] Error {}: {}\n".format(line, where, message))

if __name__ == "__main__":
    global hadError
    global hadRuntimeError
    global interpreter

    hadError = False
    hadRuntimeError = False
    interpreter = Interpreter()

    if len(sys.argv) > 2:
        print("Usage: plox [script]")
        sys.exit(64)
    if len(sys.argv) == 2:
        if sys.argv[1] == "astprint":
            expr = Binary(
                Unary(Token(TokenType.MINUS, "-", None, 1), Literal(123)),
                Token(TokenType.STAR, "*", None, 1),
                Grouping(Literal(45.67))
            )
            print(AstPrinter().print(expr))
            sys.exit(0)
        runFile(sys.argv[1])
    else:
        runPrompt()

