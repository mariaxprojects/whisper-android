package com.bpdq.whisper.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val Ink = Color(0xFF1E2A3A)
private val InkSoft = Color(0xFF3B4A5E)
private val Paper = Color(0xFFF7F4EE)
private val PaperDeep = Color(0xFFEDE7DC)
private val Amber = Color(0xFFE3A23B)
private val AmberBright = Color(0xFFF2B84B)
private val Night = Color(0xFF12161C)
private val NightSurface = Color(0xFF1A2029)
private val NightSurfaceHigh = Color(0xFF232B36)
private val Cream = Color(0xFFECE7DD)

private val LightColors = lightColorScheme(
    primary = Ink,
    onPrimary = Color.White,
    primaryContainer = PaperDeep,
    onPrimaryContainer = Ink,
    secondary = Amber,
    onSecondary = Ink,
    secondaryContainer = Color(0xFFFBE7C2),
    onSecondaryContainer = Color(0xFF4A3205),
    background = Paper,
    onBackground = Ink,
    surface = Paper,
    onSurface = Ink,
    surfaceVariant = PaperDeep,
    onSurfaceVariant = InkSoft,
    surfaceContainerLowest = Color.White,
    surfaceContainerLow = Color(0xFFFBF9F5),
    surfaceContainer = Color(0xFFF2EEE6),
    surfaceContainerHigh = PaperDeep,
    outline = Color(0xFFB9B0A2),
    outlineVariant = Color(0xFFDCD4C7),
    error = Color(0xFFB3261E),
    errorContainer = Color(0xFFF9DEDC),
    onErrorContainer = Color(0xFF410E0B),
)

private val DarkColors = darkColorScheme(
    primary = AmberBright,
    onPrimary = Ink,
    primaryContainer = NightSurfaceHigh,
    onPrimaryContainer = Cream,
    secondary = AmberBright,
    onSecondary = Ink,
    secondaryContainer = Color(0xFF4A3205),
    onSecondaryContainer = Color(0xFFFBE7C2),
    background = Night,
    onBackground = Cream,
    surface = Night,
    onSurface = Cream,
    surfaceVariant = NightSurfaceHigh,
    onSurfaceVariant = Color(0xFFB8B1A5),
    surfaceContainerLowest = Color(0xFF0D1015),
    surfaceContainerLow = NightSurface,
    surfaceContainer = NightSurface,
    surfaceContainerHigh = NightSurfaceHigh,
    outline = Color(0xFF5B6574),
    outlineVariant = Color(0xFF333C48),
)

private val AppTypography = Typography().let { t ->
    t.copy(
        headlineSmall = t.headlineSmall.copy(fontWeight = FontWeight.SemiBold, letterSpacing = (-0.2).sp),
        titleMedium = t.titleMedium.copy(fontWeight = FontWeight.SemiBold),
        bodyLarge = t.bodyLarge.copy(fontSize = 17.sp, lineHeight = 27.sp),
        labelSmall = t.labelSmall.copy(letterSpacing = 0.6.sp),
    )
}

val MonoStyle = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 18.sp)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = AppTypography,
        content = content,
    )
}
