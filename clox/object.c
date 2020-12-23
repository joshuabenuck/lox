#include <stdio.h>
#include <string.h>

#include "memory.h"
#include "object.h"
#include "value.h"
#include "vm.h"

// exists mainly to avoid the need to redundantly cast a void*
// back to the desired type
#define ALLOCATE_OBJ(type, objectType) \
    (type*)allocateObject(sizeof(type), objectType)

// 19.3 caller passes in the number of bytes to allocate which
// enables the returned pointer to be used as a pointer to a
// specific object type (and not just the base object struct)
static Obj* allocateObject(size_t size, ObjType type) {
    Obj* object = (Obj*)reallocate(NULL, 0, size);
    object->type = type;

    object->next = vm.objects;
    vm.objects = object;
    return object;
}

static ObjString* allocateString(char* chars, int length, uint32_t hash) {
    // initalize the base object structure
    ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
    string->length = length;
    string->chars = chars;
    string->hash = hash;

    return string;
}

// FNV-1a hash algorithm
static uint32_t hashString(const char* key, int length) {
    uint32_t hash = 2166136261u;

    for (int i = 0; i < length; i++) {
        hash ^= key[i];
        hash *= 16677619;
    }

    return hash;
}

ObjString* takeString(char* chars, int length) {
    uint32_t hash = hashString(chars, length);
    return allocateString(chars, length, hash);
}

ObjString* copyString(const char* chars, int length) {
    uint32_t hash = hashString(chars, length);
    char* heapChars = ALLOCATE(char, length + 1);
    memcpy(heapChars, chars, length);
    // we terminate this explicitly because the lexeme points to the middle
    // of the source string
    // this lets us pass these strings to functions in the C standard library
    // that expect a null terminated string
    heapChars[length] = '\0';

    return allocateString(heapChars, length, hash);
}

void printObject(Value value) {
    switch (OBJ_TYPE(value)) {
        case OBJ_STRING:
            printf("%s", AS_CSTRING(value));
            break;
    }
}
