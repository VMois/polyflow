#include "Parsers/SnakemakeParser.hpp"

namespace snakemake {

Token Lexer::nextToken() {
    skipWhitespace();

    if (input_.eof()) {
        return Token{END_OF_FILE, "END", line_, column_};
    }

    char c = input_.peek();
    switch (c) {
        case ':':
            input_.get();
            return Token{COLON, "COLON", line_, column_++};
        case '\t':
            input_.get();
            return Token{INDENT, "INDENT", line_, column_++};
        case '"':
            return readString();
        case '\n':
            return readNewline();
        case ',':
            input_.get();
            return Token{COMMA, "COMMA", line_, column_++};
        default:
            if (isIdentifierStart(c)) {
                return readIdentifierOrKeyword();
            } else {
                std::cerr << "Unexpected character: " << c
                        << " (line " << line_ << ", column " << column_ << ")\n";
                input_.get();
                return nextToken();
            }
    }
};

bool Lexer::isWhitespace(char c) const {
    return c == ' ';
};

void Lexer::skipWhitespace() {
    while (isWhitespace(input_.peek())) {
        input_.get();
        column_++;
    }
};

bool Lexer::isIdentifierStart(char c) const {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
};

bool Lexer::isIdentifierChar(char c) const {
    return isIdentifierStart(c) || (c >= '0' && c <= '9');
};

Token Lexer::readString() {
    std::string value;
    input_.get(); // consume the opening quote

    while (input_.peek() != '"') {
        value += input_.get();
        column_++;
    }
    input_.get(); // consume the closing quote
    return Token{STRING, value, line_, column_ - value.length()};
};

Token Lexer::readNewline() {
    input_.get(); // consume the newline
    line_++;
    column_ = 0;
    return Token{NEWLINE, "NEW_LINE", line_, column_};
};

Token Lexer::readIdentifierOrKeyword() {
    std::string value;
    while (isIdentifierChar(input_.peek())) {
        value += input_.get();
        column_++;
    }

    if (value == "rule") {
        return Token{RULE, value, line_, column_ - value.length()};
    } else if (value == "input") {
        return Token{INPUT, value, line_, column_ - value.length()};
    } else if (value == "output") {
        return Token{OUTPUT, value, line_, column_ - value.length()};
    } else if (value == "shell") {
        return Token{SHELL, value, line_, column_ - value.length()};
    } else {
        return Token{IDENTIFIER, value, line_, column_ - value.length()};
    }
};


void Parser::advance() {
    lookahead = lexer_.nextToken();
}

void Parser::throw_unexpected_token() {
    throw std::runtime_error("Unexpected token: " + lookahead.value + " on line " + std::to_string(lookahead.line) + ", column " + std::to_string(lookahead.column) + "\n");
}

void Parser::match(const TokenType tokenType) {
    if (lookahead.type == tokenType) {
        advance();
    } else {
        throw_unexpected_token();
    }
}

std::unique_ptr<ModuleAST> Parser::snakemake() {
    std::vector<std::unique_ptr<RuleAST>> rules;

    while (lookahead.type != END_OF_FILE) {
        if (lookahead.type == RULE) {
            rules.push_back(rule());
        } else if (lookahead.type == NEWLINE) {
            match(NEWLINE);
        } else {
            throw_unexpected_token();
        }
    }

    return std::make_unique<ModuleAST>(std::move(rules));
}

std::unique_ptr<RuleAST> Parser::rule() {
    match(RULE);

    std::string ruleName = "unknown";
    Token location = lookahead;

    if (lookahead.type == IDENTIFIER) {
        ruleName = lookahead.value;
        advance();
    }

    match(COLON);
    match(NEWLINE);

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string command;

    while (lookahead.type == INPUT || lookahead.type == OUTPUT || lookahead.type == SHELL) {
        if (lookahead.type == INPUT) {
            inputs = input_rule();
        } else if (lookahead.type == OUTPUT) {
            outputs = output_rule();
        } else if (lookahead.type == SHELL) {
            command = shell_rule();
        }
    }

    return std::make_unique<RuleAST>(location, ruleName, std::move(inputs), 
                                     std::move(outputs), std::move(command));
}

std::vector<std::string> Parser::input_rule() {
    match(INPUT);
    match(COLON);
    match(NEWLINE);

    return parameter_list();
}

std::vector<std::string> Parser::output_rule() {
    match(OUTPUT);
    match(COLON);
    match(NEWLINE);

    return parameter_list();
}

std::string Parser::shell_rule() {
    match(SHELL);
    match(COLON);
    match(NEWLINE);

    std::string command = lookahead.value;

    match(STRING);
    match(NEWLINE);

    return command;
}

std::vector<std::string> Parser::parameter_list() {
    std::vector<std::string> parameters;
    while (lookahead.type == STRING) {
        parameters.push_back(lookahead.value);

        match(STRING);
        if (lookahead.type == COMMA) {
            match(COMMA);
        }
        match(NEWLINE);
    }
    return parameters;
};

void dumpAST(ModuleAST &module) {
    for (auto &rule : module.rules) {
        std::cout << "rule " << rule->name << ":\n";
        std::cout << "    input:\n";
        for (auto &input : rule->inputs) {
            std::cout << "        " << input << "\n";
        }
        std::cout << "    output:\n";
        for (auto &output : rule->outputs) {
            std::cout << "        " << output << "\n";
        }
        std::cout << "    shell:\n";
        std::cout << "        " << rule->command << "\n";
    }
};


mlir::ModuleOp snakemake::MLIRGen::mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(theModule.getBody());

    for (auto &rule : moduleAST.rules) {
        mlirGen(*rule);
    }
    return theModule;
};

polyflow::StepOp snakemake::MLIRGen::mlirGen(RuleAST &rule) {
    auto name = llvm::StringRef(rule.name);

    // auto command = llvm::StringRef(rule.command);
    auto inputs = rule.inputs;
    // auto outputs = rule.outputs;

    std::vector<mlir::Value> inputValues;

    for (auto i = 0; i < inputs.size(); i++) {
        mlir::StringAttr strAttr = mlir::StringAttr::get(builder.getContext(), inputs[i]);
        auto fileLocationStr = llvm::StringRef(inputs[i]);
        polyflow::FileLocation inputOp = builder.create<polyflow::FileLocation>(builder.getUnknownLoc(), polyflow::Polyflow_StrType::get(builder.getContext()), strAttr);
        //polyflow::FileLocation inputOp = builder.create<polyflow::FileLocation>(builder.getUnknownLoc(), mlir::IntegerType::get(builder.getContext(), 69), strAttr);
        inputValues.push_back(inputOp.getResult());
    }
    
    mlir::ValueRange inputRange = mlir::ValueRange(inputValues);
    polyflow::StepOp step = builder.create<polyflow::StepOp>(
        builder.getUnknownLoc(), 
        name,
        inputRange
    );
    // mlir::Region& region = step.getBody();
    // mlir::Block *body = new mlir::Block();

    // mlir::OpBuilder::InsertionGuard guard(builder);
    // builder.setInsertionPointToStart(body);

    // region.push_back(body);

    //mlir::ArrayAttr inputsAttr = mlir::ArrayAttr::get(builder.getContext(), stringAttrs);
    //llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
    //mlir::OpBuilder::InsertPoint savedInsertPoint = builder.saveInsertionPoint();
    // Create the inputs variable using the OpBuilder
    // mlir::Value inputsVar = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(),
    //                                                        /*result type=*/inputsAttr.getType(),
    //                                                        /*value=*/inputsAttr);
    //builder.setInsertionPointToEnd(theModule.getBody());
    
    //mlir::Type result = builder.getTupleType(mlir::ArrayRef<mlir::StringAttr>());
    //builder.create<polyflow::Inputs>(builder.getUnknownLoc(), inputsAttr);
    //theModule->setAttr(llvm::StringRef(rule.name + "inputs"), inputsAttr);
    //theModule->insertRegionEntryArgument(0, inputsVar);
    //builder.restoreInsertionPoint(savedInsertPoint);

    
    // print regions
    // mlir::Region region = step.getBody();
    // mlir::Block *body = new mlir::Block();
    // body->push_back(new mlir::Block());
    // region.push_back(body);
    //return step;
};

}
