#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

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

class SnakemakeLexer {
    public:
        SnakemakeLexer(std::istream& input) : input_(input) {}
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

class TreeNode {
public:
    std::string name;
    std::vector<std::shared_ptr<TreeNode>> children;

    TreeNode(const std::string &name) : name(name) {}

    void add_child(const std::shared_ptr<TreeNode> &child) {
        children.push_back(child);
    }

    void print(int indent = 0) {
        for (int i = 0; i < indent; ++i) {
            std::cout << "  ";
        }
        std::cout << name << std::endl;
        for (const auto &child : children) {
            child->print(indent + 1);
        }
    }
};

class SnakemakeParser {
public:
    SnakemakeParser(std::istream& input): lexer_(input) {}

    void parse() {
        advance();
        snakemake();
    }

    void print_tree() {
        root->print();
    }

private:
    SnakemakeLexer lexer_;
    Token lookahead;
    std::shared_ptr<TreeNode> root = std::make_shared<TreeNode>("snakemake");

    void advance();

    void throw_unexpected_token();

    void match(const TokenType tokenType);

    std::shared_ptr<TreeNode> snakemake();
    std::shared_ptr<TreeNode> rule();

    std::shared_ptr<TreeNode> input_rule();
    std::shared_ptr<TreeNode> output_rule();
    std::shared_ptr<TreeNode> shell_rule();
    std::shared_ptr<TreeNode> parameter_list();
};
