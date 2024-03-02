<script lang="ts">
	import c from 'classnames';

	export let name = '';
	export let outlined = false;
	export let size = '16px';
	export let klass = '';

	const klasses = c(
		{
			'material-icons-outlined': outlined,
			'material-icons': !outlined
		},
		klass
	);
</script>

<span on:click on:keydown class={klasses} style:font-size={size}>{name}</span>
