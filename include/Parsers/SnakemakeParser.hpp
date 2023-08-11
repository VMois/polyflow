#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "PolyFlow/PolyFlowDialect.h"
#include "PolyFlow/PolyFlowTypes.h"
#include "PolyFlow/PolyFlowOps.h"


namespace snakemake {

enum TokenType {
    RULE,
    INPUT,
    OUTPUT,
    SHELL,
    IDENTIFIER,
    COLON,
    COMMA,
    STRING,
    NEWLINE,
    INDENT,
    END_OF_FILE,
};

struct Token {
    TokenType type;
    std::string value;
    size_t line;
    size_t column;
};

class Lexer {
    public:
        Lexer(std::istream& input) : input_(input) {}
        Token nextToken();
        bool eof() const { return input_.eof(); }
    
    private:
        std::istream& input_;
        size_t line_ = 0;
        size_t column_ = 0;

        bool isWhitespace(char c) const;

        void skipWhitespace();

        bool isIdentifierStart(char c) const;
        bool isIdentifierChar(char c) const;

        Token readString();
        Token readNewline();
        Token readIdentifierOrKeyword();
};

class BaseAST {
public:
    enum ASTKind {
        Rule,
    };

    BaseAST(ASTKind kind, Token location)
        : kind(kind), location(std::move(location)) {}
    virtual ~BaseAST() = default;

    ASTKind getKind() const { return kind; }

    const Token &loc() { return location; }

private:
    const ASTKind kind;
    Token location;
};

class RuleAST : public BaseAST {
    public:
        std::string name;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::string command;

        RuleAST(Token loc,
                std::string name,
                std::vector<std::string> inputs,
                std::vector<std::string> outputs,
                std::string command)
            : BaseAST(Rule, std::move(loc)), name(std::move(name)), inputs(std::move(inputs)), outputs(std::move(outputs)), command(std::move(command)) {}

        static bool classof(const BaseAST *c) { return c->getKind() == Rule; }
};

class ModuleAST {
    public:
        std::vector<std::unique_ptr<RuleAST>> rules;
        ModuleAST(std::vector<std::unique_ptr<RuleAST>> rules)
            : rules(std::move(rules)) {}
};

void dumpAST(ModuleAST &);

class Parser {
public:
    Parser(std::istream& input): lexer_(input) {}

    std::unique_ptr<ModuleAST> parseModule() {
        advance();
        return snakemake();
    }

private:
    Lexer lexer_;
    Token lookahead;

    void advance();

    void throw_unexpected_token();

    void match(const TokenType tokenType);

    std::unique_ptr<ModuleAST> snakemake();
    std::unique_ptr<RuleAST> rule();

    std::vector<std::string> input_rule();
    std::vector<std::string> output_rule();
    std::string shell_rule();
    std::vector<std::string> parameter_list();
};

class MLIRGen {
    public:
        MLIRGen(mlir::MLIRContext &context) : builder(&context) {}
        mlir::ModuleOp mlirGen(ModuleAST &);
    
    private:
        mlir::ModuleOp theModule;
        mlir::OpBuilder builder;

        polyflow::StepOp mlirGen(RuleAST &);
};

} // namespace snakemake
