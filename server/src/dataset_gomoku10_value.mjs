//
// dataset_gomoku10_value.mjs
// Быстрый бутстрап-датасет для Gomoku 10×10 с полноценной разметкой value.
// Терминалы, немедленные выигрыши, короткий lookahead и шаблоны.
import tfpkg from './tf.mjs';
const tf = tfpkg;

export function emptyBoard(N){ return new Int8Array(N*N).fill(0); }
const dirs = [[1,0],[0,1],[1,1],[1,-1]];
const inb = (N,r,c)=> r>=0 && c>=0 && r<N && c<N;

function checkFive(board, N, r, c, dr, dc){
  const who = board[r*N+c];
  if (who===0) return 0;
  let k=1, rr=r+dr, cc=c+dc;
  while(inb(N,rr,cc) && board[rr*N+cc]===who){ k++; rr+=dr; cc+=dc; }
  rr=r-dr; cc=c-dc;
  while(inb(N,rr,cc) && board[rr*N+cc]===who){ k++; rr-=dr; cc-=dc; }
  return (k>=5) ? who : 0;
}

export function getWinner(board, N){
  for (let r=0;r<N;r++){
    for (let c=0;c<N;c++){
      const v = board[r*N+c];
      if (!v) continue;
      for (const [dr,dc] of dirs){
        if (checkFive(board,N,r,c,dr,dc)===v) return v;
      }
    }
  }
  for (let i=0;i<board.length;i++) if (board[i]===0) return null;
  return 0;
}

function legalMoves(board){
  const m=[]; for (let i=0;i<board.length;i++) if (board[i]===0) m.push(i); return m;
}

function candidateMoves(board, N, radius=2){
  const L = board.length;
  const mark = new Uint8Array(L);
  let any=false;
  for (let i=0;i<L;i++){
    if (board[i]!==0){
      any=true;
      const r = (i/ N)|0, c = i%N;
      for (let dr=-radius;dr<=radius;dr++){
        for (let dc=-radius;dc<=radius;dc++){
          const rr=r+dr, cc=c+dc;
          if (inb(N,rr,cc)){
            const idx = rr*N+cc;
            if (board[idx]===0) mark[idx]=1;
          }
        }
      }
    }
  }
  if (!any) return [((N*N)/2)|0];
  const out=[]; for (let i=0;i<L;i++) if (mark[i]) out.push(i);
  return out.length? out : legalMoves(board);
}

function hasImmediateWin(board, N, player){
  const moves = candidateMoves(board,N,1);
  for (const mv of moves){
    board[mv] = player;
    const w = getWinner(board,N);
    board[mv] = 0;
    if (w===player) return mv;
  }
  return -1;
}

function patternScore(board, N, player){
  let open4=0, open3=0;
  for (let r=0;r<N;r++){
    for (let c=0;c<N;c++){
      for (const [dr,dc] of dirs){
        // окно 6: .XXXX.
        let seq=[], ok=true;
        for (let k=0;k<6;k++){
          const rr=r+dr*(k-1), cc=c+dc*(k-1);
          if (!inb(N,rr,cc)){ ok=false; break; }
          seq.push(board[rr*N+cc]);
        }
        if (ok){
          if (seq[0]===0 && seq[5]===0){
            let p=0,o=0; for (let t=1;t<=4;t++){ if (seq[t]===player) p++; else if (seq[t]===-player) o++; }
            if (p===4 && o===0) open4++;
          }
        }
        // окно 5: _XXX_
        seq=[]; ok=true;
        for (let k=0;k<5;k++){
          const rr=r+dr*k, cc=c+dc*k;
          if (!inb(N,rr,cc)){ ok=false; break; }
          seq.push(board[rr*N+cc]);
        }
        if (ok){
          let zeros=0,p=0,o=0;
          for (let t=0;t<5;t++){ const v=seq[t]; if (v===0) zeros++; else if (v===player) p++; else o++; }
          if (o===0 && p===3 && zeros===2) open3++;
        }
      }
    }
  }
  return { open4, open3 };
}

function quickValue(board, N, player){
  const term = getWinner(board,N);
  if (term!==null){
    if (term===player) return +1;
    if (term===-player) return -1;
    if (term===0) return 0;
  }
  const winNow = hasImmediateWin(board,N,player);
  if (winNow>=0) return +1;
  let oppWins=0;
  const candOpp = candidateMoves(board,N,1);
  for (const mv of candOpp){
    board[mv] = -player;
    const w = getWinner(board,N);
    board[mv] = 0;
    if (w===-player){ oppWins++; if (oppWins>=2) return -1; }
  }
  if (oppWins===1) return -0.2;
  const moves = candidateMoves(board,N,2);
  const scored = moves.map(mv=>{
    let s=0; const r=(mv/N)|0, c=mv%N;
    for (const [dr,dc] of dirs){
      for (let k=-2;k<=2;k++){
        const rr=r+dr*k, cc=c+dc*k;
        if (inb(N,rr,cc)){
          const v = board[rr*N+cc];
          if (v===player) s+=2; else if (v===-player) s+=1;
        }
      }
    }
    return {mv,s};
  }).sort((a,b)=>b.s-a.s).slice(0,12);

  let best = -1.0;
  for (const {mv} of scored){
    board[mv]=player;
    const w = getWinner(board,N);
    if (w===player){ board[mv]=0; return +1; }
    const oppWin = hasImmediateWin(board,N,-player);
    if (oppWin>=0){ board[mv]=0; best = Math.max(best, -0.8); continue; }
    const myPat = patternScore(board,N,player);
    const opPat = patternScore(board,N,-player);
    const h = Math.tanh( 0.4*(myPat.open4*3 + myPat.open3) - 0.4*(opPat.open4*3 + opPat.open3) );
    best = Math.max(best, h);
    board[mv]=0;
  }
  return best;
}

function toPlanes(board, current){
  const L = board.length;
  const my = new Float32Array(L), op = new Float32Array(L), empty = new Float32Array(L);
  for (let i=0;i<L;i++){
    const v = board[i];
    if (v===0) empty[i]=1; else if (v===current) my[i]=1; else op[i]=1;
  }
  return {my,op,empty};
}

export function maskFromBoard(b){ const m = new Float32Array(b.length); for(let i=0;i<b.length;i++) m[i] = b[i]===0?1:0; return m; }

export function* batchGeneratorValue({N=10, batchSize=256, steps=1000}={}){
  const L = N*N;
  for (let step=0; step<steps; step++){
    const xs = new Float32Array(batchSize*L*3);
    const ms = new Float32Array(batchSize*L);
    const ps = new Float32Array(batchSize*L);
    const vs = new Float32Array(batchSize);
    for (let b=0;b<batchSize;b++){
      const board = emptyBoard(N);
      let cur = (Math.random()<0.5)?1:-1;
      const movesToPlay = Math.floor(N*N*(0.1 + Math.random()*0.25));
      for (let t=0;t<movesToPlay;t++){
        const cand = candidateMoves(board,N,2);
        const mv = cand[(Math.random()*cand.length)|0];
        board[mv] = cur; cur = -cur;
        if (getWinner(board,N)!==null) break;
      }
      const {my,op,empty} = toPlanes(board, cur);
      const mask = maskFromBoard(board);
      const moves = legalMoves(board);
      const scores = new Float32Array(L);
      let sum=0;
      if (moves.length){
        for (const mv of moves){
          let s=0; const r=(mv/N)|0, c=mv%N;
          for (const [dr,dc] of dirs){
            for (let k=-2;k<=2;k++){
              const rr=r+dr*k, cc=c+dc*k;
              if (inb(N,rr,cc)){
                const v = board[rr*N+cc];
                if (v===cur) s+=2; else if (v===-cur) s+=1;
              }
            }
          }
          scores[mv]=s; sum+=s;
        }
      }
      const policy = new Float32Array(L);
      if (sum>0){ for (const mv of moves) policy[mv]=scores[mv]/sum; }
      else { const p = 1/Math.max(1,moves.length); for (const mv of moves) policy[mv]=p; }
      const value = quickValue(board, N, cur);
      for (let i=0;i<L;i++){
        const base = b*L*3 + i*3;
        xs[base+0]=my[i]; xs[base+1]=op[i]; xs[base+2]=empty[i];
        ms[b*L+i]=mask[i]; ps[b*L+i]=policy[i];
      }
      vs[b]=value;
    }
    yield {
      xHW3: tf.tensor4d(xs, [batchSize, N, N, 3]),
      maskFlat: tf.tensor2d(ms, [batchSize, L]),
      yPolicy: tf.tensor2d(ps, [batchSize, L]),
      yValue: tf.tensor2d(vs, [batchSize, 1]),
    };
  }
}
