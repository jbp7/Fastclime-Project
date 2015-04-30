#include <stdio.h>
#include "tree.h"
#include "myalloc.h"

#define TRUE  1
#define FALSE 0

typedef struct tnode {
    int           data;
    struct tnode *parent;
    struct tnode *left;
    struct tnode *right;
} TNODE;

static TNODE *root=NULL;

static void killnode( TNODE *node )
{
    if (node->left  != NULL) killnode(node->left);;
    if (node->right != NULL) killnode(node->right);;
    if (node != NULL) FREE(node);
}

void killtree( void )
{
    if (root != NULL) killnode(root);
    root = NULL;
  
}

void addtree( int data )
{
    TNODE *node = root;
    TNODE *parent;

    if (root == NULL) {
	MALLOC( root, 1, TNODE );
	root->data   = data;
	root->parent = NULL;
	root->left   = NULL;
	root->right  = NULL;
	return;
    }
    while ( node != NULL ) {
	     if ( node->data > data ) { parent = node; node = node->left; }
	else if ( node->data < data ) { parent = node; node = node->right; }
	else                          { return; }	
    }

    MALLOC( node, 1, TNODE );
    node->data   = data;
    node->parent = parent;
    node->left   = NULL;
    node->right  = NULL;
    if (parent->data > data) { parent->left  = node; }
    else                     { parent->right = node; }
}

void deltree( int data )
{
    TNODE *node   = root;
    TNODE *parent = NULL;
    TNODE *node1, *node2;

    while ( node != NULL && node->data != data ) {
	     if ( node->data > data ) { node = node->left; }
	else if ( node->data < data ) { node = node->right; }
    }

    if (node == NULL) { return; }  

    parent = node->parent;

    if (node->right == NULL) {
	node1 = node->left;
	if (node1 != NULL) { node1->parent = parent; }
    } else if (node->left == NULL) {
	node1 = node->right;
	node1->parent = parent;
    } else {
	node1 = node->left;
        for ( node2 = node1; node2->right != NULL; node2 = node2->right ) { }
        node2->right = node->right;
	node->right->parent = node2;
    
        node->left->parent = parent;
    }
    if (parent == NULL) {
	root = node1;
    } else {
        if (parent->data > data) { parent->left  = node1; }
	                    else { parent->right = node1; }
    }

    FREE( node );
}

static TNODE *curnode=NULL;

int getfirst( void )
{
    TNODE *node;
    TNODE *parent;

    if (root == NULL) {  return -1; }

    for (node=root; node!=NULL; parent=node, node=node->left) { }

    curnode=parent;

    return curnode->data;
}

int getnext( void )
{
    TNODE *node;
    TNODE *par;

    if (curnode==NULL) {  return -1; }

    if (curnode->right!=NULL) {
        for (node=curnode->right; node!=NULL; par=node, node=node->left) { }
	curnode = par;
	return curnode->data;
    }
    for (node=curnode->parent; node!=NULL; node=node->parent) {
	 if ( node->data > curnode->data ) break;
    }
    curnode = node;
    if (curnode!=NULL) {
	return curnode->data;
    } else {
	return -1;		
    }
}

int getlast( void )
{
    TNODE *node;
    TNODE *parent;

    if (root == NULL) {  return -1; }

    for (node=root; node!=NULL; parent=node, node=node->right) { }

    curnode=parent;

    return curnode->data;
}

int getprev( void )
{
    TNODE *node;
    TNODE *par;

    if (curnode==NULL) { return -1; }

    if (curnode->left!=NULL) {
        for (node=curnode->left; node!=NULL; par=node, node=node->right) { }
	curnode = par;
	return curnode->data;
    }
    for (node=curnode->parent; node!=NULL; node=node->parent) {
	 if ( node->data < curnode->data ) break;
    }
    curnode = node;
    if (curnode!=NULL) {
	return curnode->data;
    } else {
	return -1;		
    }
}



