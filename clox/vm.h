#ifndef clox_vm_h
#define clox_vm_h

#include "chunk.h"
#include "value.h"

#define STACK_MAX 256

typedef struct {
    Chunk* chunk;
    // Ideally would a be local variable in the VM
    // Using a pointer to an integer rather than and index
    // as it is faster than looking up an element in an
    // array by index
    uint8_t* ip;
    Value stack[STACK_MAX];
    Value* stackTop;
} VM;

typedef enum {
    INTERPRET_OK,
    INTERPRET_COMPILE_ERROR,
    INTERPRET_RUNTIME_ERROR
} InterpretResult;

void initVM();
void freeVM();
void push(Value value);
Value pop();
InterpretResult interpret(Chunk* chunk);

#endif