/* MARCH Satisfiability Solver

   Copyright (C) 2001-2009 M.J.H. Heule, J.E. van Zwieten, and M. Dufour.
   [marijn@heule.nl, jevanzwieten@gmail.com, mark.dufour@gmail.com]
  
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
  
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
  
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 

*/

#include "common.h"
#include "lookahead.h"
#include "preselect.h"
#include "equivalence.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifndef __APPLE__
  #include <malloc.h>
#endif

#define MAX_ITERATIONS		20
#define RANK_TRIGGER		0
//#define RANK_TRIGGER		100
//#define RANK_TRIGGER		10000
//#define RANK_TRIGGER		1000000

/* variables */
int *freevarsLookup;

int* CandidatesSet, nrofCandidates;

double preselect_counter;
int dynamic_preselect_setsize;

inline int compute_max_preselected();

int eq_check_flag = 1;

float *hiTmp;

void init_preselection()
{
	preselect_counter = 0;
	eq_check_flag     = 1;

	CandidatesSet = (int*  ) malloc( sizeof( int   ) * ( nrofvars + 1 ) );
        Rank          = (float*) malloc( sizeof( float ) * ( nrofvars + 1 ) );

	initial_freevars = freevars;

#ifdef HIDIFF
	clause_weight = (float*) malloc( sizeof(float) * nrofbigclauses );
#endif
	Rank_trigger = RANK_TRIGGER;
}

void dispose_preselection()
{
	FREE( CandidatesSet );
	FREE( Rank          );
#ifdef HIDIFF
	FREE( clause_weight );
#endif	
}


void init_freevars()
{
	int i, *Reductions;

	if( kSAT_flag )	Reductions = big_occ;
	else 		Reductions = TernaryImpSize;


    	freevarsArray  = (int*) malloc( sizeof( int ) * (nrofvars + 1) );
	freevarsLookup = (int*) malloc( sizeof( int ) * (nrofvars + 1) );


	MALLOC_OFFSET( hiRank, float, nrofvars, 1 );
	MALLOC_OFFSET( hiTmp,  float, nrofvars, 1 );
	MALLOC_OFFSET( hiSum,  float, nrofvars, 0 );
#ifdef HIDIFF
	if( kSAT_flag ) StaticRank( 10 );
#endif
        freevars = 0;
#ifdef EQ
//        for( i = 0; i < nrofceq; i++ )
//	    if( CeqSizes[ i ] > 2 )
//		VeqDepends[ CeqDepends[ i ] ] = EQUIVALENT; 
#endif
        for( i = 1; i <= nrofvars; i++ )
        {
	    freevarsLookup[ i ] = -1;

	    if( (Reductions[ i ] + Reductions[ -i ] == 0) &&
		(BinaryImp[  i ][ 0 ] == bImp_satisfied[  i ]) && 
		(BinaryImp[ -i ][ 0 ] == bImp_satisfied[ -i ]) &&
		(Veq[ i ][ 0 ] == 1 ) )
		    continue;

	    activevars = i;

	    if( (timeAssignments[ i ] < VARMAX) )
            {
		freevarsArray [ freevars  ] = i;
		freevarsLookup[     i     ] = freevars++;
            }
	}

	printf("c number of free variables = %i\n", freevars );
	printf("c highest active variable  = %i\n", activevars );
#ifdef CUBE
	part_free = freevars;
#endif

#ifdef DYNAMIC_PRESELECT_SETSIZE
	dynamic_preselect_setsize = check_ternary_clause_density();
#else
	dynamic_preselect_setsize = 0;
#endif
	if( dynamic_preselect_setsize ) printf("c dynamic_preselect_setsize :: on\n");
	else			  	printf("c dynamic_preselect_setsize :: off\n");
}

void dispose_freevars()
{
    	FREE( freevarsArray  );
	FREE( freevarsLookup );

	FREE_OFFSET( hiRank    );
	FREE_OFFSET( hiTmp     );
	FREE_OFFSET( hiSum     );
}

int ConstructCandidatesSet( )
{
	int i, _freevar, *_freevarsArray, *Reductions;

	if( kSAT_flag )	Reductions = big_occ;
	else		Reductions = TernaryImpSize;

	nrofCandidates = 0;

        for( _freevarsArray = freevarsArray, i = freevars; i > 0; i-- )
	{
	    _freevar = *(_freevarsArray++);

	    UNFIX(_freevar);

	    if( _freevar > original_nrofvars )                            continue;
	    if( (VeqDepends[ _freevar ] == EQUIVALENT) && eq_check_flag ) continue;
//	    if( VeqDepends[ _freevar ] != INDEPENDENT )                   continue;

#ifdef GLOBAL_AUTARKY
	    if( (depth > 0) && 
		(TernaryImpReduction[  _freevar ] == 0) &&
	        (TernaryImpReduction[ -_freevar ] == 0) )		  continue;
#endif
	    CandidatesSet[ nrofCandidates++ ] = _freevar;
	}
	return nrofCandidates;
}

void RealisePreselectedSet( )
{
        int i, lit, *_freevarsArray, max_preselected, iteration_counter;
	double som;

#ifndef LOCAL_AUTARKY

    if( dynamic_preselect_setsize ) max_preselected = compute_max_preselected();
	else				max_preselected = (int) (percent * 0.01 * freevars );
	if( max_preselected < 30 ) 	max_preselected = 30;

/*
	if     (  depth ==  0  )    max_preselected = nrofvars; else
	if     (  depth <=  2  )    max_preselected = 400; else
	if     (  depth <=  4  )    max_preselected = 200; else
	if(    (  depth <=  8  )  ||   
	 (3 * nrofceq > freevars )  )    		  // multiplier
				  max_preselected = 100;
	else if( depth <= 12 )    max_preselected =  60;
	else		          max_preselected =  30;
*/
#endif
        lookaheadArrayLength = 0;
	_freevarsArray = CandidatesSet;
        for( i = nrofCandidates, som = 0; i > 0; i-- )
	{
	    lit = *(_freevarsArray++);

        if( Rank[ lit ] > Rank_trigger )
	    {
	       	som += Rank[ lit ];
            lookaheadArray[ lookaheadArrayLength++ ] = lit;
        }
	}

	if( lookaheadArrayLength <= 5 )
	{
            lookaheadArrayLength = 0;
	    _freevarsArray = CandidatesSet;
            for( i = nrofCandidates, som = 0; i > 0; i-- )
	    {
	    	lit = *(_freevarsArray++);

	    	som += Rank[ lit ];
            	lookaheadArray[ lookaheadArrayLength++ ] = lit;
	    } 
	}

#ifndef LOCAL_AUTARKY
	iteration_counter = 0;
	while( ((lookaheadArrayLength/2) >= max_preselected) && (iteration_counter < MAX_ITERATIONS) )
	{
            double mean = (som / lookaheadArrayLength) - 1;

            for( i = 0, som = 0; i < lookaheadArrayLength; ) 
  	    	if( ( Rank[ lookaheadArray[ i ] ] ) >= mean ) 
                { 
                    som += Rank[ lookaheadArray[ i++ ] ]; 
		}
                else    lookaheadArray[ i ] = lookaheadArray[ --lookaheadArrayLength ]; 

	    assert( lookaheadArrayLength > 0 );

	    iteration_counter++;
	}
#endif

	qsort( lookaheadArray, lookaheadArrayLength, sizeof(int), RankCompare );

#ifndef LOCAL_AUTARKY
	if( lookaheadArrayLength > max_preselected )
	    lookaheadArrayLength = max_preselected;
#endif
}

void ConstructPreselectedSet( )
{
	ComputeDiffWeights();
	RealisePreselectedSet( );
}

int PreselectAll( )
{
	int i, _freevar, *Reductions;

	if( kSAT_flag )	Reductions = big_occ;
	else		Reductions = TernaryImpSize;

        lookaheadArrayLength = 0;

        for( i = 0; i < freevars; i++ )
	{
	    _freevar = freevarsArray[ i ];
	    if( (Reductions[ _freevar ] > 0) || (Reductions[ -_freevar ] > 0) ||
		(BinaryImp[  _freevar ][ 0 ] > bImp_satisfied[  _freevar ]) || 
		(BinaryImp[ -_freevar ][ 0 ] > bImp_satisfied[ -_freevar ]) ||
		(Veq[ _freevar ][ 0 ] > 1 ) )
	    {
        	lookaheadArray[ lookaheadArrayLength++ ] = _freevar;
	        CandidatesSet [ nrofCandidates++       ] = _freevar;
	    }
	}

//	printf("c preselected 'ALL' %i\n", nrofCandidates);

	ComputeDiffWeights();

	qsort( lookaheadArray, lookaheadArrayLength, sizeof( int ), RankCompare );

	return lookaheadArrayLength;
}

int RankCompare(const void *ptrA, const void *ptrB)
{
	return ( Rank[ *(int *)ptrA ] - Rank[ *(int *)ptrB ] ) > 0 ? -1 : 1;
}

int compute_max_preselected( )
{
	if( depth <= MAX_FULL_LOOK_DEPTH )
	{
	    preselect_counter += (double) freevars;
	    return freevars;
	}

	preselect_counter += (double) (5 + 7 * forced_literals );

	return (int) (preselect_counter / nodeCount);
}

void reduce_freevars( int nrval )
{
	int tmp, nr;

	if( freevarsLookup[ NR(nrval) ] == -1 ) return; 

	tmp = freevarsArray[ --freevars ];
	nr = NR(nrval);

   	freevarsArray[ freevarsLookup[ nr ] ] = tmp;
   	freevarsArray[ freevars ] = nr;

   	freevarsLookup[ tmp ] = freevarsLookup[ nr ];
   	freevarsLookup[  nr ] = freevars;

	Rank[ nr ] = 0;
}

int check_ternary_clause_density()
{
	int i, j, ternary_sum, freevars_sum;
	int *occurences;

	if( kSAT_flag )	occurences = big_occ;
	else		occurences = TernaryImpSize;

	ternary_sum 	= 0;
	freevars_sum = freevars;

        for( i = 0; i < freevars; i++ )
        {
	    j = freevarsArray[ i ];
            if ( VeqDepends[ j ] == EQUIVALENT )
              	freevars_sum--;
            ternary_sum += occurences[ j ] + occurences[ -j ];
	}

	if( freevars > freevars_sum * 1.5 ) 
	{
		printf("c many dependent variables -> full lookahead\n");
		return 1;
	}

	ternary_sum = ternary_sum / 3;

	return ((ternary_sum + nrofceq) > 3*(freevars_sum+nrofvars-original_nrofvars))? 0 : 1;
}

void StaticRank( unsigned int accuracy )
{
	int i, j, lit, *clauseSet, *literals;
	float *input, *output, weight, norm = 1, sum;

	input  = hiRank;
	output = hiTmp;


	for( i = 1; i <= nrofvars; i++ )
	{  input [  i ] = 1; input [ -i ] = 1; output[  i ] = 1; output[ -i ] = 1; }

    for( j = 0; j < 10; j++ )
    {
	for( i = 0; i < nrofbigclauses; i++ )
	{
	    weight = 1.0;
            literals = clause_list[ i ];
            while( *literals != LAST_LITERAL )
            {   lit = *(literals++); weight *= input[ -lit ]; }

            literals = clause_list[ i ];
            while( *literals != LAST_LITERAL )
            {   lit = *(literals++); output[  lit ] += weight; }
	}

	sum = 0;
	for( i = 1; i <= nrofvars; i++ )
	{
	    output[  i ] = output[  i ] / input[ -i ]; // do not count this literal
	    output[ -i ] = output[ -i ] / input[  i ]; // do not count this literal

	    sum += output[ i ] + output[ -i ];
	}

	norm = sum / (2*nrofvars);
	for( i = 1; i <= nrofvars; i++ )
	{  output[  i ] = output[  i ] / norm; output[ -i ] = output[ -i ] / norm; }

	for( i = 1; i <= nrofvars; i++ )
	{   input[  i ] = output[  i ]; input[ -i ] = output[ -i ]; output[ i ] = 1; output[ -i ] = 1; }
    }

	hiRank = input;
	hiTmp  = output;

	for( i = 0; i < nrofbigclauses; i++ )
	{
	    weight = 1;
            literals = clause_list[ i ];
            while( *literals != LAST_LITERAL )
            {   lit = *(literals++); weight *= hiRank[ -lit ]; }

//	    clause_weight[ i ] = weight;
	    clause_weight[ i ] = weight * pow(5.0, 2 - clause_length[ i ]);
	}

	for( i = 1; i <= nrofvars; i++ )
	{
	    sum = 0.1; clauseSet = clause_set[ i ];
            while( *clauseSet != LAST_CLAUSE ) { sum += clause_weight[ *(clauseSet++) ];}
	    hiSum[ -i ] = sum / hiRank[ i ];

	    sum = 0.1; clauseSet = clause_set[ -i ];
            while( *clauseSet != LAST_CLAUSE ) { sum += clause_weight[ *(clauseSet++) ];}
	    hiSum[  i ] = sum / hiRank[ -i ];
	}
}

#ifdef HIDIFF
void updateValues( unsigned int clause_index, float value )
{
	int lit, *literals = clause_list[ clause_index ];
        while( *literals != LAST_LITERAL )
	{
	    lit = (*literals++);
	    hiSum[  lit ] += value / hiRank[ -lit ];
//	    assert( hiSum[ lit ] > -0.1 );
	}
}

void HiRemoveLiteral( unsigned int clause_index, int nrval )
{
	float tmp_weight = clause_weight[ clause_index ];
	clause_weight[ clause_index ] *= 5 / hiRank[ nrval ];
	updateValues( clause_index, clause_weight[ clause_index ] - tmp_weight );
}

void HiAddLiteral( unsigned int clause_index, int nrval )
{
	float tmp_weight = clause_weight[ clause_index ];
	clause_weight[ clause_index ] *= 0.2 * hiRank[ nrval ];
	updateValues( clause_index, clause_weight[ clause_index ] - tmp_weight );
}

void HiRemoveClause( unsigned int clause_index )
{	updateValues( clause_index, -clause_weight[ clause_index ] ); }

void HiAddClause( unsigned int clause_index )
{	updateValues( clause_index,  clause_weight[ clause_index ] ); }
#endif
