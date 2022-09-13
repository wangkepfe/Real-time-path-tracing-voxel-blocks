#include "shaders/LinearMath.h"
#include "shaders/HalfPrecision.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{

INL_DEVICE Half4 FsrEasuRH(TexObj inputTex, Float2 p)
{
    ushort4 ret = tex2Dgather<ushort4>(inputTex, p.x, p.y, 0);
    ushort4ToHalf4Converter conv(ret);
    Half4 hf4 = conv.hf4;
    return hf4;
}

INL_DEVICE Half4 FsrEasuGH(TexObj inputTex, Float2 p)
{
    ushort4 ret = tex2Dgather<ushort4>(inputTex, p.x, p.y, 1);
    ushort4ToHalf4Converter conv(ret);
    Half4 hf4 = conv.hf4;
    return hf4;
}

INL_DEVICE Half4 FsrEasuBH(TexObj inputTex, Float2 p)
{
    ushort4 ret = tex2Dgather<ushort4>(inputTex, p.x, p.y, 2);
    ushort4ToHalf4Converter conv(ret);
    Half4 hf4 = conv.hf4;
    return hf4;
}

//------------------------------------------------------------------------------------------------------------------------------
// This runs 2 taps in parallel.
INL_DEVICE void FsrEasuTapH(
    Half2& aCR,
    Half2& aCG,
    Half2& aCB,
    Half2& aW,
    Half2 offX,
    Half2 offY,
    Half2 dir,
    Half2 len,
    half lob,
    half clp,
    Half2 cR,
    Half2 cG,
    Half2 cB)
{
    Half2 vX, vY;
    vX = offX * dir.xx() + offY * dir.yy();
    vY = offX * (-dir.yy()) + offY * dir.xx();
    vX *= len.x;
    vY *= len.y;
    Half2 d2 = vX * vX + vY * vY;
    d2 = min2h(d2, Half2(clp));
    Half2 wB = Half2(2.0 / 5.0) * d2 + Half2(-1.0);
    Half2 wA = Half2(lob) * d2 + Half2(-1.0);
    wB *= wB;
    wA *= wA;
    wB = Half2(25.0 / 16.0) * wB + Half2(-(25.0 / 16.0 - 1.0));
    Half2 w = wB * wA;
    aCR += cR * w;
    aCG += cG * w;
    aCB += cB * w;
    aW += w;
}

//------------------------------------------------------------------------------------------------------------------------------
// This runs 2 taps in parallel.
INL_DEVICE void FsrEasuSetH(
    Half2& dirPX,
    Half2& dirPY,
    Half2& lenP,
    Half2 pp,
    bool biST,
    bool biUV,
    Half2 lA,
    Half2 lB,
    Half2 lC,
    Half2 lD,
    Half2 lE)
{
    Half2 w = Half2(0.0);
    if (biST)
    {
        w = (Half2(1.0, 0.0) + Half2(-pp.x, pp.x) * Half2(half(1.0) - pp.y));
    }
    if (biUV)
    {
        w = (Half2(1.0, 0.0) + Half2(-pp.x, pp.x) * Half2(pp.y));
    }
    // ABS is not free in the packed FP16 path.
    Half2 dc = lD - lC;
    Half2 cb = lC - lB;
    Half2 lenX = max2h(abs(dc), abs(cb));
    lenX = rcp(lenX);
    Half2 dirX = lD - lB;
    dirPX += dirX * w;
    lenX = clamp2h(abs(dirX) * lenX);
    lenX *= lenX;
    lenP += lenX * w;
    Half2 ec = lE - lC;
    Half2 ca = lC - lA;
    Half2 lenY = max2h(abs(ec), abs(ca));
    lenY = rcp(lenY);
    Half2 dirY = lE - lA;
    dirPY += dirY * w;
    lenY = clamp2h(abs(dirY) * lenY);
    lenY *= lenY;
    lenP += lenY * w;
}

//------------------------------------------------------------------------------------------------------------------------------
//      +---+---+
//      |   |   |
//      +--(0)--+
//      | b | c |
//  +---F---+---+---+
//  | e | f | g | h |
//  +--(1)--+--(2)--+
//  | i | j | k | l |
//  +---+---+---+---+
//      | n | o |
//      +--(3)--+
//      |   |   |
//      +---+---+
__global__ void EdgeAdaptiveSpatialUpsampling(
    SurfObj outBuffer,
    // TexObj inputTex,
    SurfObj inBuffer,
    float inputViewportInPixelsX,
    float inputViewportInPixelsY,
    float inputSizeInPixelsX,
    float inputSizeInPixelsY,
    float outputSizeInPixelsX,
    float outputSizeInPixelsY)
{
    Int2 ip;
    ip.x = blockIdx.x * blockDim.x + threadIdx.x;
    ip.y = blockIdx.y * blockDim.y + threadIdx.y;

    Float4 con0;
    Float4 con1;
    Float4 con2;
    Float4 con3;

    // Output integer position to a pixel position in viewport.
    con0[0] = inputViewportInPixelsX * rcp(outputSizeInPixelsX); // 1/2
    con0[1] = inputViewportInPixelsY * rcp(outputSizeInPixelsY);
    con0[2] = float(0.5) * inputViewportInPixelsX * rcp(outputSizeInPixelsX) - float(0.5); // 1/4 - 1/2 = -1/4
    con0[3] = float(0.5) * inputViewportInPixelsY * rcp(outputSizeInPixelsY) - float(0.5);

    // // Viewport pixel position to normalized image space.
    // // This is used to get upper-left of 'F' tap.
    // con1[0] = rcp(inputSizeInPixelsX);
    // con1[1] = rcp(inputSizeInPixelsY);
    // con1[2] = float(1.0) * rcp(inputSizeInPixelsX); // Centers of gather4, first offset from upper-left of 'F'.
    // con1[3] = float(-1.0) * rcp(inputSizeInPixelsY);

    // // These are from (0) instead of 'F'.
    // con2[0] = float(-1.0) * rcp(inputSizeInPixelsX);
    // con2[1] = float(2.0) * rcp(inputSizeInPixelsY);
    // con2[2] = float(1.0) * rcp(inputSizeInPixelsX);
    // con2[3] = float(2.0) * rcp(inputSizeInPixelsY);

    // con3[0] = float(0.0) * rcp(inputSizeInPixelsX);
    // con3[1] = float(4.0) * rcp(inputSizeInPixelsY);
    // con3[2] = 0;
    // con3[3] = 0;

    // if (CUDA_CENTER_PIXEL())
    // {
    //     DEBUG_PRINT(con0);
    //     DEBUG_PRINT(con1);
    //     DEBUG_PRINT(con2);
    //     DEBUG_PRINT(con3);
    // }

    //------------------------------------------------------------------------------------------------------------------------------
    Float2 pp = Float2(ip.x, ip.y) * con0.xy + con0.zw; // * 1/2 - 1/4
    Float2 fp = floor(pp);
    pp -= fp;
    Half2 ppp = Half2(pp);

    //------------------------------------------------------------------------------------------------------------------------------
    // Float2 p0 = fp * con1.xy + con1.zw;
    // Float2 p1 = p0 + con2.xy;
    // Float2 p2 = p0 + con2.zw;
    // Float2 p3 = p0 + con3.xy;

    // if (CUDA_CENTER_PIXEL())
    // {
    //     DEBUG_PRINT(p0);
    //     DEBUG_PRINT(p1);
    //     DEBUG_PRINT(p2);
    //     DEBUG_PRINT(p3);
    // }

    // Half4 bczzR = FsrEasuRH(inputTex, p0);
    // Half4 bczzG = FsrEasuGH(inputTex, p0);
    // Half4 bczzB = FsrEasuBH(inputTex, p0);
    // Half4 ijfeR = FsrEasuRH(inputTex, p1);
    // Half4 ijfeG = FsrEasuGH(inputTex, p1);
    // Half4 ijfeB = FsrEasuBH(inputTex, p1);
    // Half4 klhgR = FsrEasuRH(inputTex, p2);
    // Half4 klhgG = FsrEasuGH(inputTex, p2);
    // Half4 klhgB = FsrEasuBH(inputTex, p2);
    // Half4 zzonR = FsrEasuRH(inputTex, p3);
    // Half4 zzonG = FsrEasuGH(inputTex, p3);
    // Half4 zzonB = FsrEasuBH(inputTex, p3);

    // if (CUDA_CENTER_PIXEL())
    // {
    //     DEBUG_PRINT(bczzR.toFloat4());
    // }

    // //------------------------------------------------------------------------------------------------------------------------------
    // Half4 bczzL = bczzB * Half4(half(0.5)) + (bczzR * Half4(half(0.5)) + bczzG);
    // Half4 ijfeL = ijfeB * Half4(half(0.5)) + (ijfeR * Half4(half(0.5)) + ijfeG);
    // Half4 klhgL = klhgB * Half4(half(0.5)) + (klhgR * Half4(half(0.5)) + klhgG);
    // Half4 zzonL = zzonB * Half4(half(0.5)) + (zzonR * Half4(half(0.5)) + zzonG);

    // half bL = bczzL.x;
    // half cL = bczzL.y;
    // half iL = ijfeL.x;
    // half jL = ijfeL.y;
    // half fL = ijfeL.z;
    // half eL = ijfeL.w;
    // half kL = klhgL.x;
    // half lL = klhgL.y;
    // half hL = klhgL.z;
    // half gL = klhgL.w;
    // half oL = zzonL.z;
    // half nL = zzonL.w;


//      +---+---+
//      | b | c |
//  +---+---+---+---+
//  | e | f | g | h |
//  +---+---+---+---+
//  | i | j | k | l |
//  +---+---+---+---+
//      | n | o |
//      +---+---+
    Int2 idxF = Int2(fp.x, fp.y);
    Int2 idxB = idxF + Int2(0, -1);
    Int2 idxC = idxF + Int2(1, -1);
    Int2 idxE = idxF + Int2(-1, 0);
    Int2 idxG = idxF + Int2(1, 0);
    Int2 idxH = idxF + Int2(2, 0);
    Int2 idxI = idxF + Int2(-1, 1);
    Int2 idxJ = idxF + Int2(0, 1);
    Int2 idxK = idxF + Int2(1, 1);
    Int2 idxL = idxF + Int2(2, 1);
    Int2 idxN = idxF + Int2(0, 2);
    Int2 idxO = idxF + Int2(1, 2);

    Half4 rgbB = Load2DHalf4<Half4>(inBuffer, idxB);
    Half4 rgbC = Load2DHalf4<Half4>(inBuffer, idxC);
    Half4 rgbE = Load2DHalf4<Half4>(inBuffer, idxE);
    Half4 rgbF = Load2DHalf4<Half4>(inBuffer, idxF);
    Half4 rgbG = Load2DHalf4<Half4>(inBuffer, idxG);
    Half4 rgbH = Load2DHalf4<Half4>(inBuffer, idxH);
    Half4 rgbI = Load2DHalf4<Half4>(inBuffer, idxI);
    Half4 rgbJ = Load2DHalf4<Half4>(inBuffer, idxJ);
    Half4 rgbK = Load2DHalf4<Half4>(inBuffer, idxK);
    Half4 rgbL = Load2DHalf4<Half4>(inBuffer, idxL);
    Half4 rgbN = Load2DHalf4<Half4>(inBuffer, idxN);
    Half4 rgbO = Load2DHalf4<Half4>(inBuffer, idxO);

    half bL = rgbB.x * half(0.5) + (rgbB.y + rgbB.z * half(0.5));
    half cL = rgbC.x * half(0.5) + (rgbC.y + rgbC.z * half(0.5));
    half iL = rgbE.x * half(0.5) + (rgbE.y + rgbE.z * half(0.5));
    half jL = rgbF.x * half(0.5) + (rgbF.y + rgbF.z * half(0.5));
    half fL = rgbG.x * half(0.5) + (rgbG.y + rgbG.z * half(0.5));
    half eL = rgbH.x * half(0.5) + (rgbH.y + rgbH.z * half(0.5));
    half kL = rgbI.x * half(0.5) + (rgbI.y + rgbI.z * half(0.5));
    half lL = rgbJ.x * half(0.5) + (rgbJ.y + rgbJ.z * half(0.5));
    half hL = rgbK.x * half(0.5) + (rgbK.y + rgbK.z * half(0.5));
    half gL = rgbL.x * half(0.5) + (rgbL.y + rgbL.z * half(0.5));
    half oL = rgbN.x * half(0.5) + (rgbN.y + rgbN.z * half(0.5));
    half nL = rgbO.x * half(0.5) + (rgbO.y + rgbO.z * half(0.5));

    // This part is different, accumulating 2 taps in parallel.
    Half2 dirPX = Half2(half(0.0));
    Half2 dirPY = Half2(half(0.0));
    Half2 lenP = Half2(half(0.0));

    FsrEasuSetH(dirPX, dirPY, lenP, ppp, true, false, Half2(bL, cL), Half2(eL, fL), Half2(fL, gL), Half2(gL, hL), Half2(jL, kL));
    FsrEasuSetH(dirPX, dirPY, lenP, ppp, false, true, Half2(fL, gL), Half2(iL, jL), Half2(jL, kL), Half2(kL, lL), Half2(nL, oL));

    Half2 dir = Half2(dirPX.x + dirPX.y, dirPY.x + dirPY.y);
    half len = lenP.x + lenP.y;

    //------------------------------------------------------------------------------------------------------------------------------
    Half2 dir2 = dir * dir;
    half dirR = dir2.x + dir2.y;
    bool zro = dirR < half(1.0 / 32768.0);
    dirR = PrxLoRsq(dirR);
    dirR = zro ? half(1.0) : dirR;
    dir.x = zro ? half(1.0) : dir.x;
    dir *= Half2(dirR);
    len = len * half(0.5);
    len = len * len;
    half stretch = (dir.x * dir.x + dir.y * dir.y) * PrxLoRcp(max1h(abs(dir.x), abs(dir.y)));
    Half2 len2 = Half2(half(1.0) + (stretch - half(1.0)) * len, half(1.0) + half(-0.5) * len);
    half lob = half(0.5) + half((1.0 / 4.0 - 0.04) - 0.5) * len;
    half clp = PrxLoRcp(lob);

    //------------------------------------------------------------------------------------------------------------------------------
    // FP16 is different, using packed trick to do min and max in same operation.
    Half2 bothR = max2h(max2h(Half2(-rgbF.x, rgbF.x), Half2(-rgbG.x, rgbG.x)), max2h(Half2(-rgbJ.x, rgbJ.x), Half2(-rgbK.x, rgbK.x)));
    Half2 bothG = max2h(max2h(Half2(-rgbF.y, rgbF.y), Half2(-rgbG.y, rgbG.y)), max2h(Half2(-rgbJ.y, rgbJ.y), Half2(-rgbK.y, rgbK.y)));
    Half2 bothB = max2h(max2h(Half2(-rgbF.z, rgbF.z), Half2(-rgbG.z, rgbG.z)), max2h(Half2(-rgbJ.z, rgbJ.z), Half2(-rgbK.z, rgbK.z)));
    // Half2 bothR = max2h(max2h(Half2(-ijfeR.z, ijfeR.z), Half2(-klhgR.w, klhgR.w)), max2h(Half2(-ijfeR.y, ijfeR.y), Half2(-klhgR.x, klhgR.x)));
    // Half2 bothG = max2h(max2h(Half2(-ijfeG.z, ijfeG.z), Half2(-klhgG.w, klhgG.w)), max2h(Half2(-ijfeG.y, ijfeG.y), Half2(-klhgG.x, klhgG.x)));
    // Half2 bothB = max2h(max2h(Half2(-ijfeB.z, ijfeB.z), Half2(-klhgB.w, klhgB.w)), max2h(Half2(-ijfeB.y, ijfeB.y), Half2(-klhgB.x, klhgB.x)));

    // This part is different for FP16, working pairs of taps at a time.
    Half2 pR = Half2(0.0);
    Half2 pG = Half2(0.0);
    Half2 pB = Half2(0.0);
    Half2 pW = Half2(0.0);

    FsrEasuTapH(pR, pG, pB, pW, Half2(0.0, 1.0) - ppp.xx(), Half2(-1.0, -1.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbB.x, rgbC.x), Half2(rgbB.y, rgbC.y), Half2(rgbB.z, rgbC.z));
    FsrEasuTapH(pR, pG, pB, pW, Half2(-1.0, 0.0) - ppp.xx(), Half2(1.0, 1.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbI.x, rgbJ.x), Half2(rgbI.y, rgbJ.y), Half2(rgbI.z, rgbJ.z));
    FsrEasuTapH(pR, pG, pB, pW, Half2(0.0, -1.0) - ppp.xx(), Half2(0.0, 0.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbF.x, rgbE.x), Half2(rgbF.y, rgbE.y), Half2(rgbF.z, rgbE.z));
    FsrEasuTapH(pR, pG, pB, pW, Half2(1.0, 2.0) - ppp.xx(), Half2(1.0, 1.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbK.x, rgbL.x), Half2(rgbK.y, rgbL.y), Half2(rgbK.z, rgbL.z));
    FsrEasuTapH(pR, pG, pB, pW, Half2(2.0, 1.0) - ppp.xx(), Half2(0.0, 0.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbH.x, rgbG.x), Half2(rgbH.y, rgbG.y), Half2(rgbH.z, rgbG.z));
    FsrEasuTapH(pR, pG, pB, pW, Half2(1.0, 0.0) - ppp.xx(), Half2(2.0, 2.0) - ppp.yy(), dir, len2, lob, clp, Half2(rgbO.x, rgbN.x), Half2(rgbO.y, rgbN.y), Half2(rgbO.z, rgbN.z));
    // FsrEasuTapH(pR, pG, pB, pW, Half2(0.0, 1.0) - ppp.xx(), Half2(-1.0, -1.0) - ppp.yy(), dir, len2, lob, clp, bczzR.xy, bczzG.xy, bczzB.xy);
    // FsrEasuTapH(pR, pG, pB, pW, Half2(-1.0, 0.0) - ppp.xx(), Half2(1.0, 1.0) - ppp.yy(), dir, len2, lob, clp, ijfeR.xy, ijfeG.xy, ijfeB.xy);
    // FsrEasuTapH(pR, pG, pB, pW, Half2(0.0, -1.0) - ppp.xx(), Half2(0.0, 0.0) - ppp.yy(), dir, len2, lob, clp, ijfeR.zw, ijfeG.zw, ijfeB.zw);
    // FsrEasuTapH(pR, pG, pB, pW, Half2(1.0, 2.0) - ppp.xx(), Half2(1.0, 1.0) - ppp.yy(), dir, len2, lob, clp, klhgR.xy, klhgG.xy, klhgB.xy);
    // FsrEasuTapH(pR, pG, pB, pW, Half2(2.0, 1.0) - ppp.xx(), Half2(0.0, 0.0) - ppp.yy(), dir, len2, lob, clp, klhgR.zw, klhgG.zw, klhgB.zw);
    // FsrEasuTapH(pR, pG, pB, pW, Half2(1.0, 0.0) - ppp.xx(), Half2(2.0, 2.0) - ppp.yy(), dir, len2, lob, clp, zzonR.zw, zzonG.zw, zzonB.zw);

    Half4 aC(pR.x + pR.y, pG.x + pG.y, pB.x + pB.y, half(0));
    half aW = pW.x + pW.y;

    //------------------------------------------------------------------------------------------------------------------------------
    Half4 pix = min4h(Half4(bothR.y, bothG.y, bothB.y, half(0)), max4h(-Half4(bothR.x, bothG.x, bothB.x, half(0)), aC * Half4(rcp(aW))));

    Store2DHalf4(pix, outBuffer, ip);
}

}